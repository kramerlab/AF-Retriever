import time
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from ast import literal_eval
import random

from tqdm.auto import tqdm
from framework import Framework
from logger import Step4SymbolCandidatesResult, Step5GroundTripletsResult, Step6VSSResult, Step6plus7VSSResult, \
    Step7VSSResult


def parse_args() -> Namespace:
    """
    Parse command‑line arguments for the experiment runner.
    Returns
    -------
    argparse.Namespace
        The populated namespace with all options.
    """
    parser = ArgumentParser(description="Run AutofocusRetriever on one of the supported datasets.",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # Experiment name
    parser.add_argument("--experiment_name", default="unnamed_experiment", help="Experiment name used to create or open the output directory.")

    # Eval. data
    parser.add_argument("--dataset", choices=['amazon', 'prime', 'mag'], required=True)
    parser.add_argument("--split", choices=["train", "val", "test", "human_generated_eval", "val-0.1",
                                            "test-0.1"], required=True, help="Dataset split to use.")
    parser.add_argument("--question_ids", default="-1", type=str, help="Question IDs to process: "
                                                                         "'-1' for all, "
                                                                         "'1-5' for range (inclusive), "
                                                                         "'[1,3,5]' for explicit list, "
                                                                         "or single ID like '42'")

    # Models
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--emb_model", default=None, help="Identifier of the large language model (LLM) to use "
                                                          "- either in the API or set in configs.json file for local access.",)

    # Optional
    parser.add_argument("--steps", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--step7_wo_step1", dest='step7_wo_step1', action='store_true',
                        help="Run step 7 (vss-based strand) skipping/ignoring step 1 (target type prediction).")
    parser.set_defaults(step7_wo_step1=False)
    parser.add_argument('--ignore_node_labels', dest='ignore_node_labels', action='store_true',
                        help="Do NOT filter entity candidates by their node types.")
    parser.add_argument('--consider_node_labels', dest='ignore_node_labels', action='store_false',
                        help="Filter entity candidates by their node types.")
    parser.set_defaults(ignore_node_labels=False)
    parser.add_argument('--ignore_edge_labels', dest='ignore_edge_labels', action='store_true',
                        help="Do NOT filter relation candidates by their edge types.")
    parser.add_argument('--consider_edge_labels', dest='ignore_edge_labels', action='store_false',
                        help="Filter relation candidates by their edge types.")
    parser.set_defaults(ignore_edge_labels=False)
    parser.add_argument('--regex_skip_triplets_w_invalid_rel_type', dest='skip_triplets_w_invalid_rel_type',
                        action='store_true')
    parser.set_defaults(skip_triplets_w_invalid_rel_type=False, help="Drop triplets with invalid, non-existing relation type in REGEX step.")
    parser.add_argument('--regex_skip_symbols_w_invalid_type', dest='skip_symbols_w_invalid_type',
                        action='store_true', help="Drop symbols with invalid type including incident relations in REGEX step.")
    parser.set_defaults(skip_symbols_w_invalid_type=False)

    parser.add_argument("--config_file_path", type=str, default=None, help="Absolute path to configuration JSON file."
                                                                           "Defaults to 'configs.json' in the root directory of AF-Retriever,"
                                                                           "i.e. the directory of this Python file (main.py).")

    # pre-computed results
    for i in range(1, 8):
        parser.add_argument(
            f"--step{i}_path",
            type=str,
            default=None,
            help=f"Absolute path to the pre‑computed result of step {i}.",

        )

    return parser.parse_args()


def qa_pair2str(experiment: Framework, q_id: int) -> str:
    """
    Print query id, query, and ground truth answers
    Args:
        experiment: Framework instance
        q_id: query id
    Returns:
        str: A text with query id, query, and ground truth answers in natural language.
    """
    query, _, answer_ids, _ = experiment.eval_data[q_id]

    out = f"\n\n++++++++++ question nr {q_id} ++++++++++++++\n"
    out += query + "\nAnswers:\n"
    expected_answers = experiment.skb_b.expected_answers(answer_ids, separator=" OR ")
    out += expected_answers[:-3]
    out += f"\n++++++++++ end of question nr {q_id} ++++++++++++++\n"
    return out


def main(question_ids: str, dataset_name: str, experiment_name: str, data_split: str, llm_model: str,
         emb_model: str, steps: list[str], ignore_node_labels: bool, ignore_edge_labels: bool,
         skip_triplets_w_invalid_rel_type: bool, skip_symbols_w_invalid_type: bool, configs_path: str = None,
         step1_path: str = None, step2_path: str = None, step3_path: str = None, step4_path: str = None,
         step5_path: str = None, step6_path: str = None, step7_path: str = None, step7_wo_step1: bool = False) -> None:
    """
    Evaluations Autofocus-Retriever on a given STaRK dataset.

    Args:
        question_ids: string with a single ID (e.g., '4'), a list (e.g., '[4,5,8]'), a range (e.g., '4-8'), or '-1' for
        all available questions for the selected split.
        dataset_name: prime, mag, or amazon
        experiment_name: experiment name determining where to store results
        data_split: train, val, test, val-0.1 for smaller validation set, or test-0.1 for smaller test set.
        llm_model: llm model name
        emb_model: embedding model name
        steps: list of steps to run
        ignore_node_labels: regards step 4, grounding
        ignore_edge_labels: regards step 4, grounding
        skip_triplets_w_invalid_rel_type: regards step 2, regular expression
        skip_symbols_w_invalid_type: regards step 2, regular expression
        configs_path: path of .json file with configurations
        step1_path: optional for pre-computed intermediate results
        step2_path: optional for pre-computed intermediate results
        step3_path: optional for pre-computed intermediate results
        step4_path: optional for pre-computed intermediate results
        step5_path: optional for pre-computed intermediate results
        step6_path: optional for pre-computed intermediate results
        step7_path: optional for pre-computed intermediate results
        step7_wo_step1: set TRUE to run step 7 (VSS) without taking target type predictions from step 1 into account.
    Returns:
        None:
    """

    framework = Framework(experiment_name, dataset_name, data_split, llm_model=llm_model, enable_vss=True,
                          emb_model=emb_model, configs_path=configs_path, steps_to_load=steps)
    # load question IDs for split:
    if question_ids == "-1":
        if data_split == "val-0.1":
            question_ids = framework.eval_data.split_indices["val"].reshape(-1).tolist()
            rng = random.Random(42)
            question_ids = rng.sample(question_ids, int(len(question_ids) * 0.1))
        else:
            question_ids = framework.eval_data.split_indices[data_split].reshape(-1).tolist()
    elif "[" in question_ids and "]" in question_ids:
        question_ids = literal_eval(question_ids)
    elif "-" in question_ids:
        start, end = question_ids.split("-")
        question_ids = range(int(start), int(end) + 1)
    else:
        question_ids = [int(question_ids)]

    # Load logger and log loaded arguments
    logger = framework.logger
    logger.log(f"Arguments: {question_ids=}, {dataset_name=}, {data_split=}, {llm_model=}, {emb_model=},"
               f"{configs_path=}, {ignore_node_labels=}, {ignore_edge_labels=}, {skip_triplets_w_invalid_rel_type=}, "
               f"{skip_symbols_w_invalid_type}, {step1_path=}, {step2_path=}, {step3_path=},"
               f"{step4_path=}, {step5_path=}, {step6_path=}, {step7_path=}, \n\n")

    # Load already computed results from files
    logger.load_step1(step1_path)
    logger.load_step2(step2_path)
    logger.load_step3(step3_path)
    logger.load_step4(step4_path)
    logger.load_step5(step5_path)
    logger.load_step6(step6_path)
    logger.load_step7(step7_path)
    logger.load_step6_and_7(step6_path)
    logger.load_step8()

    alpha = framework.settings.get("alpha")

    # main loop
    for question_id in tqdm(question_ids):
        # load and print test sample
        query, _, ground_truths, _ = framework.eval_data[question_id]
        logger.log(qa_pair2str(framework, question_id))

        if 1 in steps:  # LLM: Derive target node type
            if not question_id in logger.step1_results:
                start_time = time.time()
                target_type = framework.step1_get_target_type(question_id, query)
                step1_result = framework.validate_step1(target_type, ground_truths)
                logger.log_compute_time(question_id, 1, time.time() - start_time)
                logger.save_step1(question_id, step1_result)

        if 7 in steps:  # Pure VSS + step 6: Rank all target candidates
            if not question_id in logger.step7_results:
                if step7_wo_step1:
                    target_type = None
                else:
                    target_type = logger.get_step1_result(question_id).target_type

                step7_emb_incl_rels = framework.settings.get("step7_emb_incl_rels")
                start_time = time.time()
                top_hits, scores = framework.vss(None, query, question_id, target_type, emb_incl_rels=step7_emb_incl_rels)
                logger.log_compute_time(question_id, 7, time.time() - start_time)
                step7_result = Step7VSSResult(top_hits, scores)
                step7_result.ground_truths = ground_truths
                logger.save_step7(question_id, step7_result)

        if alpha > 0:  # steps 2 to 6 are irrelevant if alpha = 0
            if 2 in steps:  # LLM: Derive Cypher query
                if question_id not in logger.step2_results:
                    step1 = logger.get_step1_result(question_id)
                    start_time = time.time()
                    step2_result = framework.step2_derive_cypher_query(question_id, query, step1.target_type)
                    logger.log_compute_time(question_id, 2, time.time() - start_time)
                    logger.save_step2(question_id, query, step2_result)

            if 3 in steps:  # REGEX: Regular expressions - derive target node type + properties of constants + triplets incl. variables
                if question_id not in logger.step3_results:
                    step2_result = logger.get_step2_result(question_id)
                    start_time = time.time()
                    step3_result = framework.step3_regex(query, step2_result.cypher_str,
                                                         skip_triplets_w_invalid_rel_type, skip_symbols_w_invalid_type)
                    logger.log_compute_time(question_id, 3, time.time() - start_time)

                    # evaluate target node type prediction and count symbols
                    framework.validate_step3a_target_type_pred(step3_result, ground_truths)

                    if step3_result.symbols is not None:
                        framework.validate_step3b_counts(step3_result)
                    logger.save_step3(question_id, query, step3_result)

            if 4 in steps:  # VSS: Rank symbol candidates
                if not question_id in logger.step4_results:
                    step3_result = logger.get_step3_result(question_id)
                    start_time = time.time()
                    if step3_result.symbols is None:
                        step4_result = Step4SymbolCandidatesResult(skipped=True)
                        step4_result.valid_symbols = []
                    else:
                        step4_result = framework.step4_entity_search(step3_result.symbols, ignore_node_labels)
                    logger.log_compute_time(question_id, 4, time.time() - start_time)
                    logger.save_step4(question_id, query, step4_result)

            if 5 in steps:  # Set joins by intersection: Ground triplets
                if not question_id in logger.step5_results:
                    step3_result = logger.get_step3_result(question_id)
                    step4_result = logger.get_step4_result(question_id, step4_path)
                    start_time = time.time()
                    if step4_result.skipped:
                        step5_result = Step5GroundTripletsResult([set()], [], 0, 0, 0, 0, skipped=True)
                    else:

                        # align symbols
                        for uid in step3_result.symbols.keys():
                            step3_result.symbols[uid] = step4_result.valid_symbols[uid]
                        step3_result.target_variable = step4_result.valid_symbols[step3_result.target_variable.get_uid()]
                        for t in step3_result.triplets:
                            t.h = step3_result.symbols[t.h.get_uid()]
                            t.t = step3_result.symbols[t.t.get_uid()]

                        step5_result = framework.step5_ground_triplets(step3_result, ignore_node_labels,
                                                                       ignore_edge_labels, query, question_id, None, None)
                        framework.validate_step5(step5_result, ground_truths)
                    logger.log_compute_time(question_id, 5, time.time() - start_time)
                    step5_result.ground_truths = ground_truths
                    logger.save_step5(question_id, step5_result, ground_truths)

            if 6 in steps:  # VSS: Rank target candidates
                if not question_id in logger.step6_plus_7_results:
                    start_time = time.time()
                    if not question_id in logger.step6_results:
                        step3_result = logger.get_step3_result(question_id)  # for target variable
                        step5_result = logger.get_step5_result(question_id)

                        if step5_result.skipped or len(step5_result.answers_flattened) == 0:
                            fallback = True
                            logger.log("ERROR. First steps failed. Now using backup method.")
                        else:
                            fallback = False

                        if fallback:
                            step6_result = Step6VSSResult([], [], fallback)
                        else:
                            target_type = step3_result.target_type
                            target_variable = step3_result.target_variable

                            search_str = ""
                            for property_name in target_variable.properties:
                                search_str += f"{property_name}: {target_variable.properties[property_name]}; "

                            step6_emb_incl_rels = framework.settings.get("step6_emb_incl_rels")
                            top_hits, scores = framework.vss(step5_result, search_str, question_id, target_type,
                                                               emb_incl_rels=step6_emb_incl_rels)
                            step6_result = Step6VSSResult(top_hits, scores)
                        step6_result.ground_truths = ground_truths
                        logger.save_step6(question_id, step6_result)
                    else:
                        step6_result = logger.get_step6_result(question_id)
                    step7_result = logger.get_step7_result(question_id)
                    step6_and_7_result = Step6plus7VSSResult(step6_result, step7_result,
                                                             alpha, framework.settings.configs["k"])
                    logger.log_compute_time(question_id, 6, time.time() - start_time)
                    step6_and_7_result.ground_truths = ground_truths
                    logger.save_step6_plus_7(question_id, step6_and_7_result)

        if 8 in steps:  # LLM: rerank top-k candidates
            if not question_id in logger.step8_results:
                step6_plus_7_result = logger.get_step6_plus_7_result(question_id)
                start_time = time.time()
                if alpha > 0:
                    step4_result = logger.get_step4_result(question_id, step4_path)


                    all_symbol_cands = set()
                    l_max = framework.settings.get("l_max")
                    for s in step4_result.valid_symbols.values():
                        if s.candidates is not None and isinstance(s.candidates, list):
                            all_symbol_cands = all_symbol_cands.union(s.candidates[:l_max])
                    step8_result = framework.step8_llm_reranker(question_id, step6_plus_7_result, all_symbol_cands,
                                                                query)
                else:
                    step7_result = logger.get_step7_result(question_id)
                    step8_result = framework.step8_llm_reranker(question_id, step7_result, set(), query)
                logger.log_compute_time(question_id, 8, time.time() - start_time)
                framework.validate_step8(step8_result, ground_truths)
                logger.save_step8(question_id, step8_result, query)
    print('Done.')
    print(f"Results saved at {logger.output_path}.")


if __name__ == '__main__':
    """
    Evaluations Autofocus-Retriever on a given STaRK dataset.
    Run 'python main.py --help' for more information.
    """
    args = parse_args()
    main(question_ids=args.question_ids, dataset_name=args.dataset, experiment_name=args.experiment_name,
         data_split=args.split, llm_model=args.llm_model, emb_model=args.emb_model, steps=args.steps,
         ignore_node_labels=args.ignore_node_labels, ignore_edge_labels=args.ignore_edge_labels,
         skip_triplets_w_invalid_rel_type=args.skip_triplets_w_invalid_rel_type,
         skip_symbols_w_invalid_type=args.skip_symbols_w_invalid_type, configs_path=args.config_file_path,
         step1_path=args.step1_path, step2_path=args.step2_path, step3_path=args.step3_path, step4_path=args.step4_path,
         step5_path=args.step5_path, step6_path=args.step6_path, step7_path=args.step7_path,
         step7_wo_step1=args.step7_wo_step1)