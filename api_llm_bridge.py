import json
import os
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
from openai import OpenAI
import time

from logger import Logger



def prepare_chat_log(prompt: str, initial_system_message: str,
                     chat_log: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """
    Initializes chat log if it is None, and appends prompt to it
    Args:
        prompt: prompt to append
        initial_system_message: initial system message, which is only used if chat_log is None
        chat_log: optional json style chat log
    Returns: chat log including prompt and initial system message
    """
    if chat_log is None:
        if initial_system_message is None:
            chat_log = []
        else:
            chat_log = [{'role': 'system', 'content': initial_system_message}]
    chat_log.append({'role': 'user', 'content': prompt})
    return chat_log



def load_configs_from_file(file_path: str | Path = None) -> dict[str, str | int | float | bool | None]:
    if file_path is None:
        file_path = Path(__file__).parent / 'configs.json'

    with open(file_path) as json_file:
        configs = json.load(json_file)
    if "llm" in configs:
        return configs["llm"]
    return configs


class LlmBridge:

    def __init__(self, llm_model: str = None, configs_path: str | Path = None, logger: Logger=None):
        """

        Args:
            model_name:
            configs_path:
            logger:
        """
        self.logger = logger
        configs = load_configs_from_file(configs_path)

        self.llm_access_mode = configs["llm_access_mode"]  # either "api" or "local"
        if llm_model is None:
            self.model_name = configs["llm_model"]
        else:
            self.model_name = llm_model
        self.temperature = configs["llm_temperature"]
        self.seed = configs["llm_seed"]
        self.url = configs["llm_api_url"]
        self.reasoning_effort = configs["llm_reasoning_effort"]

        self.parallelization_mode = configs["llm_parallelization_mode"]
        self.initial_system_message = configs["llm_default_system_message"]

        self.do_sample = configs["llm_do_sample"]
        self.max_output_tokens = configs["llm_max_output_tokens"]

        if self.temperature is None:
            self.temperature = 0.0
            if self.model_name == "gpt-5-mini":
                self.temperature = 1.0 # minimum
        load_dotenv()
        self.client = OpenAI(api_key=os.environ.get("LLM_API_KEY"), base_url=self.url)

    def forward_to_api(self, chat_log: list[dict[str, str]]) -> tuple[str, str | None, int, int, float] | None:
        max_retries = 2
        for retries in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=chat_log,
                    max_completion_tokens=self.max_output_tokens,
                    reasoning_effort=self.reasoning_effort,  # 'low', 'medium', 'high' or 'none'
                    stream=False,
                    temperature=self.temperature,
                    n=1,
                    response_format={"type": "text"},  # { "type": "json_schema", "json_schema": {...} }
                    seed=self.seed,
                    frequency_penalty=0.05,
                    # tool_choice="none"
                )
                answer_netto = chat_completion.choices[0].message.content
                if "reasoning_content" in chat_completion.choices[0].message.model_extra:
                    reasoning_content = chat_completion.choices[0].message.model_extra["reasoning_content"]
                else:
                    reasoning_content = None
                if "</think>" in answer_netto:
                    if not isinstance(reasoning_content, str):
                        reasoning_content = ""
                    reasoning_content += answer_netto.split("</think>")[0]
                    answer_netto = answer_netto.split("</think>")[-1]
                if "boxed{" in answer_netto and answer_netto.rfind("}") != -1:
                    if not isinstance(reasoning_content, str):
                        reasoning_content = ""
                    reasoning_content += answer_netto.split("boxed{")[0]
                    answer_netto = answer_netto[:answer_netto.rfind("}")].split("boxed{")[-1]

                num_input_tokens = chat_completion.usage.prompt_tokens
                num_output_tokens = chat_completion.usage.completion_tokens
                try:
                    estimated_costs = chat_completion.usage.model_extra["estimated_cost"]
                except KeyError:
                    estimated_costs = 0.0

                return answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs

            except RuntimeError as e:
                raise e
        return None

    def forward_to_api_batch_inner(self, idx: int, answers_netto: list[str], num_input_tokens_list: list[str],
                                   num_output_tokens_list: list[str], estimated_costs_list: list[str], chat_log: list[dict[str, str]]):
        answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs = self.forward_to_api(chat_log)
        answers_netto[idx] = answer_netto
        num_input_tokens_list[idx] = num_input_tokens
        num_output_tokens_list[idx] = num_output_tokens
        estimated_costs_list[idx] = estimated_costs

    def ask_llm(self, question: str, chat_log=None, log: bool = True, qid:int = None, step:int = None) -> tuple[str, str | None, int, int, float]:
        chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_log)
        answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs = self.forward_to_api(chat_log)
        if qid is not None:
            self.logger.log_llm_costs(qid, step, num_input_tokens=num_input_tokens, num_output_tokens=num_output_tokens,
                                      estimated_costs=estimated_costs)

        if log and self.logger is not None:
            log_str = f"\n[Ask Question]: {question}\n\n"
            if reasoning_content is not None:
                log_str += f"[{self.model_name} Reasoning Content]: {reasoning_content}\n"
            log_str += f"{self.model_name} Answer]: {answer_netto}\n\n"
            self.logger.log(log_str)
        return answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs

    def ask_llm_batch(self, questions: list[str], chat_logs:list[list[dict[str,str]]] = None, log: bool = False):
        if self.parallelization_mode == "batch_processing":
            raise ValueError("Batch processing is currently not supported for API calls. "
                             "Use parallelization_mode 'multiprocessing' instead.")

        if self.logger is not None:
            self.logger.log(f"\n[LLM Query 1/{len(questions)} in batch]: {questions[0]}\n")

        if chat_logs is None:
            chat_logs = [None for _ in range(len(questions))]

        if self.parallelization_mode == "sequential":
            answers_netto, reasoning_contents, num_input_tokens_list, num_output_tokens_list, estimated_costs_list = [], [], [], [], []
            for i in range(len(questions)):
                answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs = self.ask_llm(questions[i], chat_logs[i])
                answers_netto.append(answer_netto)
                reasoning_contents.append(reasoning_content)
                num_input_tokens_list.append(num_input_tokens)
                num_output_tokens_list.append(num_output_tokens)
                estimated_costs_list.append(estimated_costs)
        else:
            procs = []
            answers_netto = [None for _ in range(len(questions))]
            num_input_tokens_list = [None for _ in range(len(questions))]
            num_output_tokens_list = [None for _ in range(len(questions))]
            estimated_costs_list = [None for _ in range(len(questions))]

            for idx, question in enumerate(questions):
                chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_logs[idx])
                chat_logs[idx] = chat_log
                p = Thread(target=self.forward_to_api_batch_inner, args=(idx, answers_netto, num_input_tokens_list,
                                                                    num_output_tokens_list, estimated_costs_list, chat_log))
                procs.append(p)
                p.start()
            for p in procs:
                p.join()
            for i, answer in enumerate(answers_netto):
                chat_logs[i].append({"role": "assistant", "content": answer})

        if self.logger is not None:
            self.logger.log(f"[{self.model_name} Answer]: {answers_netto[0]}\n")
        return answers_netto, num_input_tokens_list, num_output_tokens_list, estimated_costs_list


def test_llm_bridge():
    llm = LlmBridge()

    print(f"+++ Test with {llm.model_name} +++")

    chat_log = [
        {"role": "system", "content": "Respond like a michelin starred chef."},
        {"role": "user", "content": "Can you name at least two different techniques to cook lamb?"},
        {"role": "assistant",
         "content": "Bonjour! Let me tell you, my friend, cooking lamb is an art form, and I'm more than happy to "
                    "share with you not two, but three of my favorite techniques to coax out the rich, unctuous "
                    "flavors and tender textures of this majestic protein. First, we have the classic \"Sous Vide\" "
                    "method. Next, we have the ancient art of \"Sous le Sable\". "
                    "And finally, we have the more modern technique of \"Hot Smoking.\""},
        {"role": "user",
         "content": "Tell me more about the second method. "},
    ]
    chat_log = [
        {"role": "system", "content": "Answer briefly in machine-readable format."},
        {"role": "user", "content": "What is two plus one?"},
    ]
    start_time = time.time()
    answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs = llm.forward_to_api(chat_log)

    print(f"Time elapsed: {time.time() - start_time:.1f} seconds\n")

    print(answer_netto)
    print("Input and output tokens: ", num_input_tokens, num_output_tokens)
    print(f"Estimated costs: {estimated_costs:.2f}€ ({estimated_costs:.2e})€")
    print("Reasoning content: ", reasoning_content)


if __name__ == "__main__":
    test_llm_bridge()