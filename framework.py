import datetime
import shutil
import regex as re

from stark_qa import load_skb

from llm_reranker import LLMReranker
from regex_parser import parse_cypher_to_triplets, parse_conditions_from_cypher
from stark_qa.load_qa import load_qa
from optional.load_qa_offline import load_qa_offline
from stark_qa.skb import SKB

import api_llm_bridge
import local_llm_bridge
from logger import *
from skb_bridge import SKBbridge
from settings import Settings
from triplet import TripletEnd, Triplet



class Framework:
    def __init__(self, experiment_name: str, dataset_name: str, data_split: str, llm_model: str = None, skb: SKB = None,
                 enable_vss: bool = True, emb_model: str = None, configs_path: str = None, steps_to_load: list[str] = None):
        # plausibility checks
        valid_dataset_names = ['prime', 'mag', 'amazon']
        if dataset_name not in valid_dataset_names:
            raise ValueError(f"Dataset {dataset_name} not found. It should be in {valid_dataset_names}")

        valid_data_splits = ["train", "val", "test", "human_generated_eval", "test-0.1", "val-0.1"]
        if data_split not in valid_data_splits:
            raise ValueError(f"Data split {data_split} not found. It should be in {valid_data_splits}")

        # load settings
        self.settings = Settings(dataset_name, llm_model=llm_model, emb_model=emb_model, configs_path=configs_path)

        llm_model = self.settings.get("llm")["llm_model"]
        configs = self.settings.configs

        # load logger
        abs_full_output_path = Path(__file__).parent / configs["output_path"] / dataset_name / data_split / llm_model / experiment_name
        self.logger = Logger(abs_full_output_path)

        # copy config file to results file
        new_configs_path = abs_full_output_path
        if (abs_full_output_path / self.settings.configs_path.parts[-1]).exists():
            new_configs_path /= (self.settings.configs_path.parts[-1] + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"))
        shutil.copy(self.settings.configs_path, new_configs_path)

        # load llm bridge
        if configs["llm"]["llm_access_mode"] == "api":
            self.llm_bridge = api_llm_bridge.LlmBridge(llm_model, self.settings.configs_path, self.logger)
        else:
            self.llm_bridge = local_llm_bridge.LlmBridge(llm_model, self.settings.configs_path, self.logger)

        # load SKB bridge including SKB, embeddings, and embedding client
        if skb is None:
            if self.settings.get("skb_path") == "auto_download":
                skb = load_skb(name=dataset_name, download_processed=False)
            else:
                skb = load_skb(name=dataset_name, download_processed=True, root=self.settings.get("skb_path"))
        load_embs_w_rels = False
        load_embs_wo_rels = False
        if enable_vss and steps_to_load is not None:
            if 4 in steps_to_load:
                if self.settings.get("step4_emb_incl_rels"):
                    load_embs_w_rels = True
                else:
                    load_embs_wo_rels = True
            if 6 in steps_to_load:
                if self.settings.get("step6_emb_incl_rels"):
                    load_embs_w_rels = True
                else:
                    load_embs_wo_rels = True
            if 7 in steps_to_load:
                if self.settings.get("step7_emb_incl_rels"):
                    load_embs_w_rels = True
                else:
                    load_embs_wo_rels = True
        if load_embs_w_rels or load_embs_wo_rels:
            emb_dir = Path(__file__).parent / configs["embeddings_path"] / dataset_name
        else:
            emb_dir = None
        self.skb_b = SKBbridge(settings=self.settings, data_split=data_split, llm_bridge=self.llm_bridge, skb=skb,
                               emb_dir=emb_dir, load_embs_w_rels=load_embs_w_rels, load_embs_wo_rels=load_embs_wo_rels)

        # load data

        if configs["offline_mode"]:
            loading_function = load_qa_offline
        else:
            loading_function = load_qa
        self.eval_data = loading_function.__call__(name=dataset_name, human_generated_eval=data_split=="human_generated_eval",
                                     root=self.settings.get("qa_path"))

        # load reranker
        self.reranker = LLMReranker(skb, self.llm_bridge, self.skb_b.vss, self.settings, self.logger)

        print(f"Size of test dataset: {len(self.eval_data)} QA-pairs")


    def step1_get_target_type(self, qid: int, question: str) -> str:
        """
        Prompts an LLM to return the type of entities that are searched for in a given question.
        Args:
            qid (int):
            question (str):
        Returns: Type of searched entity or entities
        """

        candidate_types = self.skb_b.skb.candidate_types

        # If only one candidate type is available return it immediately and skip the prompting.
        if len(candidate_types) == 1:
            return candidate_types[0]
        else:
            llm_query = (f"Given several instances of these types: {candidate_types}. An instance of which type could "
                         f"correctly answer the query: {question}\n\n"
                         f"Return nothing but the type of which the instance must be of. Think straightforwardly.")

            target_type, _, num_input_tokens, num_output_tokens, expected_costs = self.llm_bridge.ask_llm(llm_query, qid=qid, step=1)
            if target_type is not None:
                target_type = target_type.strip("'\" \n[]")
            return target_type

    def validate_step1(self, target_type: str, ground_truths: list[int]) -> Step1PredictTargetTypeResult:
        candidate_types = self.skb_b.skb.node_type_lst() # self.skb_b.skb.candidate_types
        ground_truth_node_type = self.skb_b.skb.get_node_type_by_id(ground_truths[0])

        is_invalid = False
        is_incorrect = False

        target_type = target_type.split(",")[0]
        if target_type is None or target_type not in candidate_types:
            is_invalid = 1
            self.logger.log(f"Target type {target_type} is not a valid target type."
                            f"It must be in {candidate_types}.")
            target_type = None
        elif target_type != ground_truth_node_type:
            is_incorrect = 1

        r = Step1PredictTargetTypeResult(target_type, is_invalid, is_incorrect, ground_truth_node_type)
        return r

    def step2_derive_cypher_query(self, qid: int, query: str, target_type = None) -> Step2DeriveCypherResult:
            nodes_to_consider = str(self.settings.configs.get("nodes_to_consider"))  #self.skb_b.skb.node_type_lst()).replace("'","")

            nodes_to_consider_str = ""
            try:
                n = 5
                for nt in ast.literal_eval(nodes_to_consider):
                    ns = self.skb_b.skb.get_node_ids_by_type(nt)
                    nodes_to_consider_str += nt + " (e.g., "
                    nodes_to_consider_str += ", ".join([self.skb_b.skb.node_info[ns[int(i * 1 / n * len(ns)) - 1]]["name"] for i in range(n)])
                    nodes_to_consider_str += "), "
                nodes_to_consider_str = nodes_to_consider_str[:-2]
            except KeyError:
                nodes_to_consider_str = str(nodes_to_consider)

            edges_to_consider = str(list(self.settings.configs.get("edge_type_long2short").keys())).replace("'","")
            prompt = ("Generate a Cypher query based on the given query Q. Please follow the restrictions precisely! \n\n"
                       "* Simple Syntax: Use a very basic and short Cypher syntax.\n"
                       "* Content Accuracy: Omit any information that cannot be exactly captured with one of the given node labels, or available keywords. "
                      "Any node attributes are allowed.\n"
                       "* No Quantifications: Avoid using quantifications.\n"
                       '* No Negations: Skip negated facts, avoid using "NOT" or "<>".\n'
                       '* No "OR": Do not use "OR".\n'
                       '* Available Keywords: Restrict yourself to the available keywords: MATCH, WHERE, RETURN, AND.\n'
                       '* Date Format: Format dates as YYYY-MM-DD.\n\n'
                       'Given Information: \n'
                       f'* Query Q: {query}\n\n'
                       f'* Available Node Labels: {nodes_to_consider_str}\n'
                       f'* Available Relationship Labels: {edges_to_consider}\n'
                       f'Example: MATCH (d:disease)-[:is_effect/phenotype_of_disease]->(e:effect/phenotype)\n'
                      f'MATCH (e)-[:protein/gene_is_associated_with_effect/phenotype]->(g:gene/protein) \nWHERE g.name = "IGF1" and g.molecular_weight=120 \nRETURN d.title\n\n')
            if target_type is not None:
                prompt += f'At the end of the query, RETURN y.title for the target (y:{target_type})\n'
            prompt += f'Only return one Cypher query, no additional information.'

            cypher_str, _, num_input_tokens, num_output_tokens, expected_costs = self.llm_bridge.ask_llm(prompt, qid=qid, step=2)
            return Step2DeriveCypherResult(cypher_str)

    def step3_regex(self, query: str, cypher_str: str,  skip_triplets_w_invalid_rel_type: bool,
        skip_symbols_w_invalid_type: bool) -> Step3RegexResult:

        target_var_not_in_triplets = False
        if cypher_str is None:
            return Step3RegexResult(None, None, None, None,
                                    "ERROR: Cypher string is None.",
                                    target_var_not_in_triplets)

        rel_dict = self.settings.configs.get("edge_type_long2short")
        properties_dict = self.settings.configs.get("node_properties_dict")
        node_type_list = self.skb_b.skb.node_type_lst()

        cypher_str = cypher_str.split("[FINAL ANSWER:]")[-1].replace("[FINAL ANSWER]", "")
        cypher_str = cypher_str.strip("´`\n ;")
        cypher_str_split = cypher_str.split("RETURN")
        if len(cypher_str_split) != 2:
            return Step3RegexResult(None, None, None, None,
                                    "ERROR: Cypher string does not contain exactly one RETURN operation.",
                                    target_var_not_in_triplets)

        match_part, return_part = cypher_str_split

        triplets, symbols = parse_cypher_to_triplets(match_part, rel_dict, properties_dict, node_type_list,
                                                     skip_triplets_w_invalid_rel_type, skip_symbols_w_invalid_type)
        parse_conditions_from_cypher(match_part, symbols, properties_dict)

        target_var_name = return_part.split(".")[0].strip()
        target_type_pattern = re.compile(r'\b' + target_var_name + r'\b:([^\s)]+)')
        target_type = re.findall(target_type_pattern, cypher_str)
        if len(target_type) == 0:
            return Step3RegexResult(None, None, None, None,
                                    "ERROR: No target type in Cypher string found. Using None.", target_var_not_in_triplets)
        warnings = ""
        target_type = target_type[0]
        if target_type not in node_type_list:
            warnings += f"WARNING: Target type {target_type} not in node type list."
            target_type = None
        if target_var_name not in symbols:
            symbols[target_var_name] = TripletEnd(target_var_name, target_type, is_constant=False)
            warnings += "WARNING: Target variable not in triplets."
            target_var_not_in_triplets = True
        else:
            symbols[target_var_name].node_type = target_type

        t_variable = symbols[target_var_name]
        t_variable.is_constant = False
        symbols_w_uid = {}
        for symbol in symbols.values():
            # if symbol.node_type is not None:
            symbols_w_uid[symbol.get_uid()] = symbol

        self.logger.log(warnings)
        if "title" not in t_variable.properties and "name" not in t_variable.properties:
            t_variable.properties["title"] = query

        step3_result = Step3RegexResult(target_type, triplets, symbols_w_uid, t_variable, warnings, target_var_not_in_triplets)
        return step3_result

    def validate_step3a_target_type_pred(self, step3result: Step3RegexResult, ground_truths: list[int]):
        candidate_types = self.skb_b.skb.node_type_lst()
        ground_truth_node_type = self.skb_b.skb.get_node_type_by_id(ground_truths[0])

        is_invalid = False
        is_incorrect = False

        if  step3result.target_type not in candidate_types:
            is_invalid = 1
            self.logger.log(f"Target type {step3result.target_type} is not a valid target type."
                            f"It must be in {candidate_types}.")
            step3result.target_type = None
        elif  step3result.target_type != ground_truth_node_type:
            is_incorrect = 1
        step3result.set_target_type_pred(step3result.target_type, is_invalid, is_incorrect, ground_truth_node_type)

    def validate_step3b_counts(self, step3result: Step3RegexResult):
        constants = {}
        num_valid_constants = 0
        num_valid_variables = 0
        for symbol in step3result.symbols.values():
            if symbol.is_constant:
                num_valid_constants += 1
                constants[symbol.get_uid()] = symbol
            else:
                num_valid_variables += 1

        step3result.num_valid_constants = num_valid_constants
        step3result.num_valid_variables = num_valid_variables


    def step4_entity_search(self, valid_symbols: dict[str, TripletEnd], ignore_node_labels: bool) -> Step4SymbolCandidatesResult:
        r = Step4SymbolCandidatesResult()

        invalid_symbols = []

        for symbol in valid_symbols.values():
            candidates = set()
            if symbol.node_type is None:
                target_name = ""
            else:
                target_name = f"type: {symbol.node_type}; "

            for property_name in symbol.properties:
                property_val = symbol.properties[property_name]
                if property_name in self.settings.configs.get("node_properties_dict").keys():
                    property_name = self.settings.configs.get("node_properties_dict")[property_name]
                if property_name == "title" or property_name == "name":
                    target_name += f"{property_name}: {property_val}; "
                else:
                    new_candidates = []
                    if property_val[0] == "<" or property_val[0] == ">":
                        operator = property_val[0]
                        netto_val = property_val[1:].strip()
                        if netto_val[0] == "=":
                            operator += "="
                            netto_val = netto_val[1:]
                        try:
                            netto_val = float(netto_val)
                            if symbol.node_type is None:
                                node_ids = self.skb_b.nodes_alias2id_unknown_type
                            else:
                                node_ids = self.skb_b.node_ids_by_type[symbol.node_type]
                            for c in node_ids:
                                if property_name in self.skb_b.skb.node_info[c]:
                                    c_property_val = float(self.skb_b.skb.node_info[c][property_name])
                                    if operator == "<":
                                        if c_property_val < netto_val:
                                            new_candidates.append(c)
                                    elif operator == "<=":
                                        if c_property_val <= netto_val:
                                            new_candidates.append(c)
                                    elif operator == ">":
                                        if c_property_val > netto_val:
                                            new_candidates.append(c)
                                    elif operator == ">=":
                                        if c_property_val > netto_val:
                                            new_candidates.append(c)
                                try:
                                    if "details" in self.skb_b.skb.node_info[c] and property_name in self.skb_b.skb.node_info[c]["details"]:
                                        c_property_val = float(self.skb_b.skb.node_info[c]["details"][property_name])
                                        if operator == "<":
                                            if c_property_val < netto_val:
                                                new_candidates.append(c)
                                        elif operator == "<=":
                                            if c_property_val <= netto_val:
                                                new_candidates.append(c)
                                        elif operator == ">":
                                            if c_property_val > netto_val:
                                                new_candidates.append(c)
                                        elif operator == ">=":
                                            if c_property_val >= netto_val:
                                                new_candidates.append(c)
                                except TypeError as e:
                                    self.logger.log(str(e))
                        except ValueError: # values are not ints or floats
                            pass

                    else:
                        try:
                            new_candidates = self.skb_b.skb.get_node_ids_by_value(symbol.node_type, property_name, property_val)
                            new_candidates += self.skb_b.skb.get_node_ids_by_value(
                                symbol.node_type, property_name, int(property_val))
                        except ValueError:
                            pass
                    new_candidates = set(new_candidates)

                    if len(new_candidates) > 0:
                        if len(candidates) == 0:
                            candidates = new_candidates
                        else:
                            candidates.intersection_update(new_candidates)
                    else:
                        # if property search was not successful, add it to the embedding search string
                        target_name += f"{property_name}: {property_val}; "

                    self.logger.log(f"Number of nodes with matching alias for {property_name} found in database: {len(candidates)}.")

            if symbol.is_constant:
                if ignore_node_labels:
                    target_type = None
                else:
                    target_type = symbol.node_type
                candidates_sorted = self.skb_b.find_closest_nodes_w_cutoff(
                    target_name=target_name,
                    target_type=target_type,
                    logger=self.logger,
                    enable_vss=self.settings.get("vss_cutoff") < 1.0,
                    cutoff_vss=self.settings.get("vss_cutoff"),
                    l_max = self.settings.get("l_max"),
                    emb_incl_rels=self.settings.get("step4_emb_incl_rels")
                )
                if len(candidates) == 0:
                    candidates = candidates_sorted
                else:
                    candidates = [x for x in candidates_sorted if x in candidates]

            if len(candidates) > 0:
                self.logger.log(f"Entities found for {symbol.name}::{symbol.node_type}:"
                               f"{len(candidates) > 10 =}, {list(candidates)[:10]=},\n"
                                    f"candidate names: {self.skb_b.entity_ids2name(candidates, n=10)}")
                symbol.candidates = candidates

        for symbol_key in invalid_symbols:
            valid_symbols.pop(symbol_key)

        r.valid_symbols = valid_symbols
        return r

    def step5_ground_triplets(self, step3_result: Step3RegexResult,ignore_node_labels: bool, ignore_edge_labels: bool,
                              query: str, qid: int, answers_so_far, answers_flattened_so_far) -> Step5GroundTripletsResult:
        if answers_so_far is None:
            answers = [set[int]()]
            answers_flattened = []
        else:
            answers = answers_so_far
            answers_flattened = answers_flattened_so_far
        l_first_hit, l_last_hit = 0, 0
        num_variables_without_candidates, num_variable_candidates = 0, 0
        variables_in_use = set()
        variables_in_use.add(step3_result.target_variable)
        cnt = 0

        # identify variables that are connected to target node
        while cnt != len(variables_in_use):
            cnt = len(variables_in_use)
            for triplet in step3_result.triplets:
                if triplet.h in variables_in_use:
                    variables_in_use.add(triplet.t)
                if triplet.t in variables_in_use:
                    variables_in_use.add(triplet.h)
        vars_not_connected_to_target = [v.get_uid() for v in (set(step3_result.symbols.values()) - variables_in_use)]
        self.logger.log(f"Variables connected to target node: \n{[v.get_uid() for v in variables_in_use]}\n"
                        f"Variables not connected to target node: \n{vars_not_connected_to_target}\n")


        candidate_clones = {}
        for a in step3_result.symbols.values():
            candidate_clones[a.get_uid()] = a.candidates

        l = 1
        while l <= self.settings.configs["l_max"]:
            for a in step3_result.symbols.values():
                if a.is_constant:
                    if type(candidate_clones[a.get_uid()]) is list:
                        a.candidates = set(candidate_clones[a.get_uid()][:l])
                    else:
                        a.candidates = candidate_clones[a.get_uid()]
                else:
                    a.candidates = candidate_clones[a.get_uid()]
                    if isinstance(a.candidates, list):
                        a.candidates = set(a.candidates)
            target_variable = step3_result.target_variable
            target_variable.candidates = candidate_clones[target_variable.get_uid()]
            if isinstance(step3_result.target_variable.candidates, list):
                target_variable.candidates = set(target_variable.candidates)
            answers_l, num_variables_without_candidates, num_variable_candidates, target_variable_used = (
                self.step5_inner_grounding(step3_result.triplets, step3_result.symbols, target_variable,
                                           ignore_node_labels, ignore_edge_labels, variables_in_use))

            answers_l = set(answers_l) - set(answers_flattened)

            answers.append(answers_l)
            answers_flattened.extend(list(answers_l))
            l = int(l * 1.5 + 0.5)

            if len(answers_flattened) > 0 and l_first_hit == 0:
                l_first_hit = l
            if len(answers_flattened) >= self.settings.configs["k"]:
                l_last_hit = l
                break
            if not target_variable_used:
                self.logger.log("Target variable not used.")
                break


        step5_result = Step5GroundTripletsResult(answers, answers_flattened, num_variables_without_candidates,
                                                 num_variable_candidates, l_first_hit, l_last_hit, skipped=False)
        return step5_result



    def step5_inner_grounding(self, triplets: list[Triplet], symbols: dict[str, TripletEnd], target_variable: TripletEnd,
                        ignore_node_labels: bool, ignore_edge_labels: bool, variables_in_use: set[TripletEnd]) -> [set[int], int, int, set[str]]:
        logger = self.logger

        num_variables_without_candidates = 0
        num_variable_candidates = 0

        symbols = self.skb_b.ground_triplets(triplets, symbols, logger, target_variable, ignore_edge_labels,
                                         ignore_node_labels, variables_in_use)

        logger.log(f"Candidates for symbol terms:")
        target_variable_used = True
        for symbol_uid in symbols.keys():
            if symbols[symbol_uid].candidates is None:
                logger.log(f"{symbol_uid}: Variable not used.")
                if symbol_uid == target_variable.get_uid():
                    target_variable_used = False
            else:
                num_cands = len(symbols[symbol_uid].candidates)
                limit = 50
                if num_cands > limit:
                    logger.log(f"{symbol_uid}: More than {limit} ({num_cands}) candidates found.")
                else:
                    logger.log(f"{symbol_uid}: {symbols[symbol_uid].candidates}")
                if not symbols[symbol_uid].is_constant:
                    if num_cands == 0:
                        num_variables_without_candidates += 1
                    num_variable_candidates += num_cands

        answer = target_variable.candidates
        if answer is None:
            answer = set()
        logger.log(f"{len(answer)=}\n10 answers from the candidates set:\n"
                  f"{self.skb_b.entity_ids2name(answer, 10)}\n\n")
        return answer, num_variables_without_candidates, num_variable_candidates, target_variable_used

    def validate_step5(self, r: Step5GroundTripletsResult, ground_truths: list[int]):
        # Stats
        for gt in ground_truths:
            if gt in r.answers_flattened:
                r.num_true_pos_in_prefilter += 1

        r.num_target_candidates = len(r.answers_flattened)
        r.num_false_pos_in_prefilter = r.num_target_candidates - r.num_true_pos_in_prefilter

        r.recall = r.num_true_pos_in_prefilter / len(ground_truths)
        if r.num_true_pos_in_prefilter == 0:
            r.precision = 0.0
        else:
            r.precision = r.num_true_pos_in_prefilter / r.num_target_candidates



        self.logger.log(f"\nStep 5:\n Number of true positives: {r.num_true_pos_in_prefilter},"
                        f" number of false positives: {r.num_false_pos_in_prefilter}")



    def vss(self, step5_result: Step5GroundTripletsResult | None, query: str, query_id: int, target_type: str | None,
            emb_incl_rels = True):
        filtered_candidates = None
        if step5_result is not None:
            filtered_candidates = step5_result.answers_flattened
        top_k_node_ids, vss_scores = self.skb_b.vss.get_top_k_nodes(search_str=query, k=self.skb_b.skb.num_candidates,
                                                                    node_type=target_type, logger=self.logger,
                                                                    node_id_mask=filtered_candidates,
                                                                    complement_with_non_masked_ids=True,
                                                                    query_id=query_id,
                                                                    node_types_to_consider=self.settings.get("nodes_to_consider"),
                                                                    emb_incl_rels=emb_incl_rels)
        # sort answers by the time when they were found
        if step5_result is not None:
            new_order = []
            for current_set in step5_result.answers:
                for element in top_k_node_ids:
                    if element in current_set:
                        new_order.append(element)
            top_hits_vss = new_order
        else:
            top_hits_vss = top_k_node_ids

        return top_hits_vss[:self.settings.get("k")], vss_scores[:self.settings.get("k")]


    def step8_llm_reranker(self, qid: int, step6and7_result: Step6plus7VSSResult, node_id_mask: set[int], query: str):

        top_hits_vss = step6and7_result.vss_top_hits
        top_hits = self.reranker.rerank(qid, top_hits_vss, query, node_id_mask=node_id_mask)

        self.logger.log(f"Results (IDs): {top_hits=}")
        top_hits_str = str([self.skb_b.entity_id2name(x) for x in top_hits])
        self.logger.log(f"Results (aliases): {top_hits_str}")
        return Step8FinalRerankerResult(top_hits, top_hits_str)

    def validate_step8(self, step8_result: Step8FinalRerankerResult, ground_truths: list[int]):
        step8_result.ground_truth_str = self.skb_b.entity_ids2name(ground_truths, 10)
        step8_result.ground_truths = ground_truths