import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from logger import Logger


def load_local_llm(model_path, tokenizer_path):
    """
    Loads local LLM with the transformers package
    Args:
        model_path: absolute or relative path to model files
        tokenizer_path: absolute or relative path to corresponding tokenizer files
    Returns: loaded model and tokenizer

    """
    print(f"Loading tokenizer from {tokenizer_path}.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True,
                                              dtype="bfloat16", padding_side="left")

    tokenizer.pad_token = tokenizer.eos_token
    print(f"Loading pretrained model from {model_path}.")
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, device_map="auto",
                                                 dtype="bfloat16", low_cpu_mem_usage=False, trust_remote_code=True)
    model.eval()
    return model, tokenizer



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


def local_inference_pipeline(model_name: str, model, tokenizer, query: str, chat_log: list[dict[str,str]],
                             initial_system_message: str, max_output_tokens: int, temperature:bool = None,
                             do_sample:bool = False, top_p:bool = None, boxed: bool = False, reasoning_effort = None)\
        -> tuple[str, str, int, int, float, list[dict[str, str]]]:
    if "r1_distill" in model_name:
        initial_system_message = None   # recommended according to Usage Recommendations on
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#usage-recommendations and
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        if boxed:
            query += " Put your final answer within \\boxed{}."

    chat_log = prepare_chat_log(query, initial_system_message, chat_log=chat_log)
    inputs = tokenizer.apply_chat_template(chat_log, add_generation_prompt=True, return_tensors="pt",
                                           reasoning_effort=reasoning_effort).to(model.device)
    input_ids = inputs.input_ids
    input_size = input_ids.size(1)
    attention_mask = inputs.attention_mask

    if input_size > tokenizer.model_max_length:
        answer = "The input sequence is too long. Aborting."
        #chat_log.append({'role': 'assistant', 'content': answer})
        raise OverflowError(answer)

    #attention_mask = torch.ones(1, input_size).to(model.device)

    output = model.generate(
        input_ids,
        max_length=input_size + max_output_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,  # Handle padding gracefully
        attention_mask=attention_mask,
        do_sample=do_sample,
        top_p=top_p
    )

    if (output[0][-1] != tokenizer.eos_token_id
            and len(output[0]) - input_size == max_output_tokens):
        print(f"WARNING: Max. token length ({max_output_tokens}) exceeded.")

    answer = tokenizer.decode(output[0][input_size:], skip_special_tokens=True).strip()
    # full_answer = answer
    reasoning_content = ""
    if "r1_distill" in model_name:
        answer = answer.split("</think>")[-1]
        reasoning_content = answer.split("</think>")[0]
        if "boxed{" in answer and answer.rfind("}") != -1:
            answer = answer[:answer.rfind("}")].split("boxed{")[-1]
        answer = answer.split("Answer:**")[-1]
        answer = answer.replace("\_", "_").replace("\'", "'")
        if len(answer.split("**")) == 3:
            answer = answer.split("**")[-2]
        answer = answer.strip()
    if "gpt-oss" in model_name:
        answer = answer.split("assistantfinal")[-1]
        reasoning_content = answer.split("assistantfinal")[0]
    chat_log.append({'role': 'assistant', 'content': answer})
    return answer, reasoning_content, input_size, len(output[0]) - input_size, 0.0, chat_log
    # return answer, chat_log, full_answer


def local_inference_pipeline_batch(model_name: str, model, tokenizer, queries: list[str],
                   system_message: str, max_output_tokens: int, chat_logs: list[list[dict[str, str]]], temperature:bool = None,
                   do_sample:bool = False, top_p:bool = None, boxed: bool = False, reasoning_effort = None) -> tuple[
    list[str], list[list[dict[str, str]]], list[str]]:
    if "r1_distill" in model_name:
        # system_message = None   # recommended according to Usage Recommendations on
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B#usage-recommendations and
        # https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
        if boxed:
            for i in range(len(queries)):
                queries[i] +=  " Put your final answer within \\boxed{}."
    if chat_logs is None:
        chat_logs = [None for _ in range(len(queries))]
    for i in range(len(queries)):
        chat_logs[i] = prepare_chat_log(queries[i], system_message, chat_log=chat_logs[i])

    outputs = []

    prompts = tokenizer.apply_chat_template(chat_logs, add_generation_prompt=True, return_tensors="pt",
                                            padding=True, truncation=False, tokenize=False, reasoning_effort=reasoning_effort)
    tokenized_input = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to('cuda')

    num_input_tokens = tokenized_input.input_ids.size(1)
    batch_size = int(tokenizer.model_max_length / 2 / num_input_tokens)
    if batch_size < 0 and num_input_tokens <= tokenizer.model_max_length:
        batch_size = 1
    input_batches = torch.split(tokenized_input.input_ids, batch_size)
    attention_batches = torch.split(tokenized_input.attention_mask, batch_size)

    for i in range(len(input_batches)):
        batch_output = model.generate(
            input_batches[i],
            max_length=num_input_tokens + max_output_tokens,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_batches[i],
            do_sample=do_sample,
            top_p=top_p
        )
        outputs.extend(batch_output)

    # Decode responses and update chat logs
    for i in range(len(outputs)):
        outputs[i] = outputs[i][num_input_tokens:]

    full_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    answers = []
    for i, answer in enumerate(full_answers):
        if "r1_distill" in model_name:
            answer = answer.split("</think>")[-1]
            if "boxed{" in answer and answer.rfind("}") != -1:
                answer = answer[:answer.rfind("}")].split("boxed{")[-1]
            answer = answer.split("Answer:**")[-1]
            answer = answer.replace("\_", "_").replace("\'", "'")
            if len(answer.split("**")) == 3:
                answer = answer.split("**")[-2]
        answer = answer.strip()
        chat_logs[i].append({'role': 'assistant', 'content': answer})
        answers.append(answer)

    return answers, chat_logs, full_answers


def load_configs_from_file(file_path: str | Path = None) -> dict[str, str | int | float | bool | None]:
    if file_path is None:
        file_path = Path(__file__).parent / 'configs.json'

    with open(file_path) as json_file:
        configs = json.load(json_file)
    if "llm" in configs:
        return configs["llm"]
    return configs


class LlmBridge:
    def __init__(self, model_name: str, configs_path: str | Path = None, logger: Logger=None):
        """

        Args:
            model_name:
            configs_path:
            logger:
        """
        self.logger = logger
        configs = load_configs_from_file(configs_path)

        self.llm_access_mode = configs["llm_access_mode"]  # either "api" or "local"
        self.model_name = model_name
        self.temperature = configs["llm_temperature"]
        self.seed = configs["llm_seed"]

        self.parallelization_mode = configs["llm_parallelization_mode"]
        self.initial_system_message = configs["llm_default_system_message"]

        self.do_sample = configs["llm_do_sample"]
        self.top_p = configs["llm_top_p"]
        self.max_output_tokens = configs["llm_max_output_tokens"]
        self.reasoning_effort = configs["llm_reasoning_effort"]


        model_path = configs[self.model_name + "_path"]
        tokenizer_path = configs[self.model_name + "_path"]
        self.model, self.tokenizer = load_local_llm(model_path, tokenizer_path)



    def ask_llm(self, question: str, chat_log=None, log: bool = True, qid:int = None, step:int = None) -> tuple[str, str | None, int, int, float]:
        chat_log = prepare_chat_log(question, self.initial_system_message, chat_log=chat_log)
        answer_netto, reasoning_content, num_input_tokens, num_output_tokens, estimated_costs, _ = local_inference_pipeline(self.model_name, self.model, self.tokenizer,
                                                                 question, chat_log, self.initial_system_message,
                                                                 self.max_output_tokens, self.temperature,
                                                                 self.do_sample, self.top_p, True, self.reasoning_effort)
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

    def ask_llm_batch(self, questions: list[str], chat_logs:list[dict[str,str]] =None):
        if self.llm_access_mode == "api" and self.parallelization_mode == "batch_processing":
            raise ValueError("Batch processing is currently not supported for API calls. "
                             "Use parallelization_mode 'multiprocessing' instead.")
        elif self.llm_access_mode == "local" and self.parallelization_mode == "multiprocessing":
            raise ValueError("Multi-threading is not supported for local LLMs. "
                             "Use parallelization mode 'batch_processing' instead.")

        if self.logger is not None:
            self.logger.log(f"\n[LLM Query 1/{len(questions)} in batch]: {questions[0]}\n")

        if self.parallelization_mode == "sequential":
            if chat_logs is None:
                chat_logs = [None for _ in range(len(questions))]
            answers, chat_logs_new, full_answers = [], [], []
            for i in range(len(questions)):
                answer, chat_log, full_answer = self.ask_llm(questions[i], chat_logs[i], log=False)
                answers.append(answer)
                full_answers.append(full_answer)
                chat_logs[i] = chat_log
            self.logger.log(f"[{self.model_name} Full Answer]: {full_answers[0]}\n")
        else:
            answers, chat_logs, full_answers = local_inference_pipeline_batch(self.model_name, self.model,
                                                                                  self.tokenizer, questions,
                                                                                  self.initial_system_message,
                                                                                  self.max_output_tokens, chat_logs,
                                                                                  self.temperature, self.do_sample,
                                                                                  self.top_p, True, self.reasoning_effort)
            self.logger.log(f"[{self.model_name} Full Answer]: {full_answers[0]}\n")
        if self.logger is not None:
            self.logger.log(f"[{self.model_name} Shortened Answer]: {answers[0]}\n")
        return answers, chat_logs
