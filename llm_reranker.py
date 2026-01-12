import openai
import torch
from typing import Union, List

from stark_qa.skb import SKB

from logger import Logger
from settings import Settings
from vss import VSS

import functools  # This is for the custom sorting using python's built-in sorting methods


class LLMReranker:
    def __init__(self,
                 skb,
                 llm_bridge,
                 vss: VSS,
                 settings: Settings,
                 logger: Logger,
                 sim_weight: float = 0.0):
        """
        Initializes the LLMReranker model.

        Args:
            skb (SemiStruct): Knowledge base.
            llm_model : The actual LLM model.
            tokenizer : The actual tokenizer.
            emb_model_name (str): Embedding model name.
            sim_weight (float): Weight for similarity score.
        """
        self.skb: SKB = skb
        self.parent_vss = vss
        self.logger = logger
        self.settings = settings
        self.llm_bridge = llm_bridge
        self.sim_weight = sim_weight
        self.reranking_method = self.settings.get("reranking_method")
        self.max_k = settings.get("k")

        self.add_rel = settings.configs["llm"]["add_rel"]
        self.compact_docs = settings.configs["llm"]["compact_docs"]

    def pointwise_reranking(self, qid: int, top_k_node_ids: list[int], query: Union[str, List[str]], all_symbol_cands: set[int],
                          add_rel: bool, reduce_rel=False) -> (list[int]):
        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:

        """
        cand_len = len(top_k_node_ids)

        pred_dict = {}

        prompts = []
        answers = []
        for idx, node_id in enumerate(top_k_node_ids):
            node_type = self.skb.get_node_type_by_id(node_id)
            if reduce_rel:
                doc_info = self.skb.get_doc_info(node_id, add_rel=False, compact=self.compact_docs)
                doc_info += self.relations_2hop_2str(all_symbol_cands, node_id) + "\n"
            else:
                doc_info = self.skb.get_doc_info(node_id, add_rel=add_rel, compact=self.compact_docs)
                doc_info = str(node_id) + " " + doc_info + "\n"

            prompts.append(
                f'Examine if a {node_type} '
                f'satisfies a given query and assign a score from 0.0 to 1.0. '
                f'If the {node_type} does not satisfy the query, the score should be 0.0. '
                f'If there exists explicit and strong evidence supporting that {node_type} '
                f'satisfies the query, the score should be 1.0. If partial evidence or weak '
                f'evidence exists, the score should be between 0.0 and 1.0.\n'
                f'Here is the query:\n\"{query}\"\n'
                f'Here is the information about the {node_type}:\n' +
                doc_info + '\n\n' +
                f'Please score the {node_type} based on how well it satisfies the query. '
                f'ONLY output the floating point score WITHOUT anything else. '
                f'Output: The numeric score of this {node_type} is: '
            )
            output, _, _, _, _ = self.llm_bridge.ask_llm(prompts[-1], qid=qid, step=8)
            output = output.split(":")[-1]
            answers.append(output)
        #answers, _ = self.llm_bridge.ask_llm_batch(prompts, chat_logs=None)

        for idx, node_id in enumerate(top_k_node_ids):
            try:
                llm_score = float(answers[idx])
            except TypeError:
                if answers[idx] is None:
                    if add_rel:
                        raise RuntimeError()
                    else:
                        llm_score = 0.5
            except ValueError:
                llm_score = 0.5
            sim_score = (cand_len - idx) / cand_len
            score = llm_score + self.sim_weight * sim_score

            # prefer nodes that have been in node_id_mask, i.e. that have been prefiltered
            #if node_id_mask is not None:
            #    score /= 2
            #    if idx < len(node_id_mask):
            #        score += 0.5
            pred_dict[node_id] = score

        node_scores = torch.FloatTensor(list(pred_dict.values()))
        top_k_idx = torch.topk(node_scores, min(self.max_k, len(node_scores)), dim=-1, largest=True, sorted=True
                               ).indices.tolist()
        top_k_node_ids = [list(pred_dict.keys())[i] for i in top_k_idx]

        return top_k_node_ids

    def listwise_reranking(self, qid: int, top_k_node_ids: list[int],
                          query: Union[str, List[str]],
                          all_symbol_cands: set[int], add_rel: bool, reduce_rel: bool = False) -> (list[int]):

        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:
            a ordered list by how well the elements satsify the given query.
        """

        def method1_for_list_of_nodes(node_ids_to_rerank, query):

            if len(node_ids_to_rerank) == 0:
                return []

            possible_answers = ""

            for node_id in node_ids_to_rerank:
                if reduce_rel:
                    doc_info = self.skb.get_doc_info(node_id, add_rel=False, compact=self.compact_docs)
                    possible_answers += str(node_id) + " " + doc_info + self.relations_2hop_2str(all_symbol_cands, node_id) + "\n"
                else:
                    doc_info = self.skb.get_doc_info(node_id, add_rel=add_rel, compact=self.compact_docs)
                    possible_answers += str(node_id) + " " + doc_info + "\n"



            prompt = (
                f'The rows of the following list consist of an ID number, a type and a corresponding descriptive text:\n'
                f'{possible_answers} \n'
                f'Please sort this list in descending order according to how well the elements can be considered as '
                f'answers to the following query: \n'
                f'{query} \n'
                f'Please make absolutely sure that the element which satisfies the query best is the first element in your order. '
                f'Return ONLY the corresponding ID numbers separated by commas in the asked order.'
            )

            #output, _ = self.llm_bridge.ask_llm_batch([prompt], chat_logs=None)
            output, _, _, _, _ = self.llm_bridge.ask_llm(prompt, qid=qid, step=8)

            try:
                output = output.strip("'\" \n[]")
                answer = [int(node_id_str.strip()) for node_id_str in output.split(",")]

                answer = list(dict.fromkeys(answer))  # Remove duplicate Node_ids

                sorted_IDs = [node_id for node_id in answer if node_id in node_ids_to_rerank]  # remove invented IDs
                invented_ids = len(answer) - len(sorted_IDs)
                print("LLM has invented: ", invented_ids, " node IDs in it's answer.")
                missing_ids = len(node_ids_to_rerank) - len(sorted_IDs)
                print("LLM out does not contain ", missing_ids, " IDs from the input.")

            except:
                sorted_IDs = []
                print("LLM output contains elements that cannot be cast to integer.")
                print("Erroneous LLM output: ", output)

            sorted_IDs += [node_id for node_id in node_ids_to_rerank if node_id not in sorted_IDs]

            return sorted_IDs

        to_rerank = top_k_node_ids

        answer = method1_for_list_of_nodes(to_rerank, query)

        #answer_prioritize_prefiltered = [x for x in answer if x in node_id_mask]
        #answer_prioritize_prefiltered += [x for x in answer if x not in node_id_mask]
        #return answer_prioritize_prefiltered
        return answer

    def pairwise_comparison(self, qid: int, node1_id: int, node2_id: int, query: str, all_symbol_cands: set[int],
                            add_rel: bool, reduce_rel: bool = False):
        """
        Function to compare two nodes in a SKB by how good they satisfy a query.

        Args:
            node1_id: ID of the first node
            node2_id: ID of the second node
            query: Query
        Returns:
            {-1,0,1} depending on:
            -1 if node2_id satisfies the given query better.
            0 if the LLM output cannot be cast to a node_ID (in many cases the LLM outputs neither) or it is none of the given node IDs or if the two node_ids are identical.
            1 if node1_id satisfies the given query better.
        """

        if node1_id == node2_id:  # make sure that the comparison is reflective.
            return 0

        node_type_1 = self.skb.get_node_type_by_id(node1_id)
        node_type_2 = self.skb.get_node_type_by_id(node2_id)

        if reduce_rel:
            doc_info_1 = self.skb.get_doc_info(node1_id, add_rel=False, compact=self.compact_docs)
            doc_info_2 = self.skb.get_doc_info(node2_id, add_rel=False, compact=self.compact_docs)
            doc_info_1 += self.relations_2hop_2str(all_symbol_cands, node1_id)
            doc_info_2 += self.relations_2hop_2str(all_symbol_cands, node2_id)
        else:
            doc_info_1 = self.skb.get_doc_info(node1_id, add_rel=add_rel, compact=self.compact_docs)
            doc_info_2 = self.skb.get_doc_info(node2_id, add_rel=add_rel, compact=self.compact_docs)


        prompt = (
            f'The following two elements consist of an ID number, a type and a corresponding descriptive text:\n \n'
            f'ID number: {node1_id}, {node_type_1}, {doc_info_1}. \n'
            f'ID number: {node2_id}, {node_type_2}, {doc_info_2}. \n\n'
            f'Find out which of the elements satisfies the following query better: \n'
            f'{query} \n'
            f'Return ONLY the corresponding ID number which corresponds to the element that satisfies '
            f'the given query best. Nothing else.'
        )

        answer, _, _, _, _ = self.llm_bridge.ask_llm(prompt, qid=qid, step=8)
        if isinstance(answer, str):
            answer = answer.replace("'", "").replace('"', '').strip("'\" \n[]")
        if answer == "A":
            answer = node1_id
        elif answer == "B":
            answer = node2_id

        try:
            answer = int(answer)
        except:
            self.logger.log("LLM output cannot be cast to int.")
            self.logger.log(f"Erroneous LLM output: , {answer}, {prompt}")
            return 0  # we then assume the elements to be equally bad/good - often output is neither satisfies the query?
        if answer == node1_id:
            return 1
        elif answer == node2_id:
            return -1
        else:
            self.logger.log("LLM output is neither of the given node IDs")
            self.logger.log(f"Erroneous LLM output: , {answer}, {prompt}")
            return 0

    def rel_to_str(self, h: int, t: int, e: str, include_head_id: bool = False):

        t_name = ""
        if "title" in self.skb.node_info[t]:
            t_name = self.skb.node_info[t]["title"]
        elif "name" in self.skb.node_info[t]:
            t_name = self.skb.node_info[t]["name"]

        h_name = ""
        if "title" in self.skb.node_info[h]:
            h_name = self.skb.node_info[h]["title"]
        elif "name" in self.skb.node_info[h]:
            h_name = self.skb.node_info[h]["name"]
        if include_head_id:
            h_name = f"{h} ({h_name})"
        return f"{h_name} - {e} -> {t_name}"

    def pairwise_reranking(self, qid: int, top_k_node_ids: list[int], query: Union[str, List[str]],
                           all_symbol_cands: set[int], add_rel: bool, reduce_rel: bool = False) -> (list[int]):
        """
        Forward pass to compute predictions for the given query using LLM reranking.

        Args:
            query (Union[str, list]): Query string or a list of query strings.
            query_id (Union[int, list, None]): Query index (optional).

        Returns:
            a ordered list by how well the elements satisfy the given query.
        """

        to_rerank = top_k_node_ids

        answer = sorted(to_rerank, key = functools.cmp_to_key(lambda node1_id, node2_id : self.pairwise_comparison(
                   qid, node1_id, node2_id, query = query, all_symbol_cands=all_symbol_cands, add_rel=add_rel, reduce_rel=reduce_rel)), reverse = True)
        return answer

    def rerank(self, qid:int, top_k_node_ids: list[int], query: Union[str, List[str]], node_id_mask: set) -> (list[int]):

        top_k_node_ids = top_k_node_ids[:self.max_k]
        add_rel = self.add_rel

        try:
            if self.reranking_method == "pointwise":
                sorted_node_ids = self.pointwise_reranking(qid, top_k_node_ids, query, node_id_mask, add_rel=add_rel)

            elif self.reranking_method == "listwise":
                sorted_node_ids = self.listwise_reranking(qid, top_k_node_ids, query, node_id_mask, add_rel=add_rel)

            elif self.reranking_method == "pairwise":
                sorted_node_ids = self.pairwise_reranking(qid, top_k_node_ids, query, node_id_mask, add_rel=add_rel)
            else:
                raise (NotImplementedError("Reranking_method_not_specified!"))
        except openai.BadRequestError as e:
            if e.status_code == 400 and ("Requested input length" in e.message or "max_tokens must be at least 1" in e.message) and add_rel:
                self.logger.log("Input length for LLM exceeded. Removing relations that were not found from document descriptions now.")
                try:
                    if self.reranking_method == "pointwise":
                        sorted_node_ids = self.pointwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                   add_rel=add_rel, reduce_rel=True)

                    elif self.reranking_method == "listwise":
                        sorted_node_ids = self.listwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                  add_rel=add_rel, reduce_rel=True)

                    elif self.reranking_method == "pairwise":
                        sorted_node_ids = self.pairwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                  add_rel=add_rel, reduce_rel=True)
                    else:
                        raise (NotImplementedError("Reranking_method_not_specified!"))
                except openai.BadRequestError as e2:
                    if e.status_code == 400 and ("Requested input length" in e.message or "max_tokens must be at least 1" in e.message) and add_rel:
                        try:
                            self.logger.log("Input length for LLM exceeded. Removing all relations from document descriptions now.")
                            if self.reranking_method == "pointwise":
                                sorted_node_ids = self.pointwise_reranking(qid, top_k_node_ids, query, node_id_mask, add_rel=False)

                            elif self.reranking_method == "listwise":
                                sorted_node_ids = self.listwise_reranking(qid, top_k_node_ids, query, node_id_mask, add_rel=False)

                            elif self.reranking_method == "pairwise":
                                sorted_node_ids = self.pairwise_reranking(qid, top_k_node_ids, query, node_id_mask, add_rel=False)
                            else:
                                raise (NotImplementedError("Reranking_method_not_specified!"))
                        except openai.BadRequestError as e3:
                            if e.status_code == 400 and ("Requested input length" in e.message or "max_tokens must be at least 1" in e.message):
                                self.logger.log("Input length for LLM still exceeded. Skipping reranking now.")
                                sorted_node_ids = top_k_node_ids
                            else:
                                raise e3
                    else:
                        raise  e2
            else:
                raise  e
        except OverflowError as e:
            if "The input sequence is too long. Aborting." == e.args[0] and add_rel:
                self.logger.log(
                    "Input length for LLM exceeded. Removing relations that were not found from document descriptions now.")
                try:
                    if self.reranking_method == "pointwise":
                        sorted_node_ids = self.pointwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                   add_rel=add_rel, reduce_rel=True)

                    elif self.reranking_method == "listwise":
                        sorted_node_ids = self.listwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                  add_rel=add_rel, reduce_rel=True)

                    elif self.reranking_method == "pairwise":
                        sorted_node_ids = self.pairwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                  add_rel=add_rel, reduce_rel=True)
                    else:
                        raise (NotImplementedError("Reranking_method_not_specified!"))
                except OverflowError as e2:
                    if "The input sequence is too long. Aborting." == e.args[0] and add_rel:
                        try:
                            self.logger.log(
                                "Input length for LLM exceeded. Removing all relations from document descriptions now.")
                            if self.reranking_method == "pointwise":
                                sorted_node_ids = self.pointwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                           add_rel=False)

                            elif self.reranking_method == "listwise":
                                sorted_node_ids = self.listwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                          add_rel=False)

                            elif self.reranking_method == "pairwise":
                                sorted_node_ids = self.pairwise_reranking(qid, top_k_node_ids, query, node_id_mask,
                                                                          add_rel=False)
                            else:
                                raise (NotImplementedError("Reranking_method_not_specified!"))
                        except OverflowError as e3:
                            if "The input sequence is too long. Aborting." == e.args[0]:
                                self.logger.log("Input length for LLM still exceeded. Skipping reranking now.")
                                sorted_node_ids = top_k_node_ids
                            else:
                                raise e3
                    else:
                        raise e2
            else:
                raise e

        return sorted_node_ids

    def relations_2hop_2str(self, all_symbol_cands : set[int], target: int):
        rels_str = ""

        for edge_type in self.skb.edge_type_dict.values():
            neighbors_1hop = self.skb.get_neighbor_nodes(target, edge_type)
            for i, neighbor in enumerate(neighbors_1hop):
                if neighbor in all_symbol_cands:
                    rels_str_chain = self.rel_to_str(target, neighbor, edge_type, include_head_id=True)
                    for edge_type_2nd_hop in self.skb.edge_type_dict.values():
                        neighbors_next_hop = self.skb.get_neighbor_nodes(neighbor, edge_type_2nd_hop)
                        for next_neighbor in neighbors_next_hop:
                            if next_neighbor in all_symbol_cands and next_neighbor not in neighbors_1hop and next_neighbor != target:
                                rels_str_chain += ", " + self.rel_to_str(neighbor, next_neighbor, edge_type_2nd_hop)
                    rels_str += rels_str_chain + "\n"
        if rels_str == "":
            "No potentially relevant relations found.\n"

        else:
            rels_str = "Potentially relevant relations:\n" + rels_str
        return rels_str

