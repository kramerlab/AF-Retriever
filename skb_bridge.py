from pathlib import Path

import torch
from stark_qa.skb import SKB

from api_llm_bridge import LlmBridge
from logger import Logger
from settings import Settings
from triplet import Triplet, TripletEnd
from vss import VSS


def add_node_to_node_dict(node_dict: dict[str, int|list[int]], node_alias: str, node_id: int) -> None:
    node_alias = node_alias.lower()
    if node_alias in node_dict:
        if isinstance(node_dict[node_alias], list):
            node_dict[node_alias].append(node_id)
        else:
            node_dict[node_alias] = [node_dict[node_alias], node_id]
    else:
        node_dict[node_alias] = node_id


def create_node_dict_prime(skb: SKB, nodes_alias2id: dict[str, dict[str,int|list[int]]]):
    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        n_type = node["type"]
        n_name = node['name']
        add_node_to_node_dict(nodes_alias2id[n_type], n_name, i)

        if 'details' in node:
            if 'alias' in node['details']:
                alias = node['details']['alias']
                if isinstance(alias, list):
                    for a in alias:
                        add_node_to_node_dict(nodes_alias2id[n_type], a, i)
                else:
                    add_node_to_node_dict(nodes_alias2id[n_type], alias, i)
    return nodes_alias2id


def create_node_dict_mag(skb: SKB, nodes_alias2id: dict[str, dict[str,int|list[int]]]):
    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        n_type = node["type"]
        if 'title' in node:
            add_node_to_node_dict(nodes_alias2id[n_type], node['title'], i)
        elif 'DisplayName' in node and node['DisplayName'] != -1 and node['DisplayName'] != "-1":
            add_node_to_node_dict(nodes_alias2id[n_type], node['DisplayName'], i)
    return nodes_alias2id


def create_node_dict_amazon(skb: SKB, nodes_alias2id: dict[str, dict[str,int|list[int]]]):
    for i in range(skb.num_nodes()):
        node = skb.node_info[i]
        if 'title' in node:
            n_type = "product"
            name = node['title']
        elif 'brand_name' in node:
            n_type = "brand"
            name = node['brand_name']
        elif 'category_name' in node:
            n_type = "category"
            name = node['category_name']
        else:
            n_type = "color"
            name = node['color_name']

        add_node_to_node_dict(nodes_alias2id[n_type], name, i)
    return nodes_alias2id


def create_node_dict(skb: SKB, dataset: str) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """
    Build alias->ID mappings for all node types in a knowledge base.

    The function inspects a semi‑structured knowledge base (`skb`) and returns two
    dictionaries that allow conversion from a human‑readable *alias* (e.g. an
    author name) to the internal numeric node identifier(s).

    The first dictionary groups the mappings by node type (e.g. ``'author'``,
    ``'paper'``) while the second collapses every node type into a single global
    namespace.

    Args:
        skb: semi-structured knowledge base
        dataset: name of dataset and semi-structured knowledge base
    """
    nodes_alias2id = {}
    for n_type in skb.node_type_lst():
        nodes_alias2id[n_type] = {}

    if dataset == 'prime':
        nodes_alias2id = create_node_dict_prime(skb, nodes_alias2id)
    elif dataset == 'mag':
        nodes_alias2id = create_node_dict_mag(skb, nodes_alias2id)
    elif dataset == 'amazon':
        nodes_alias2id = create_node_dict_amazon(skb, nodes_alias2id)
    else:
        raise ValueError(f"dataset name should be in ['prime', 'mag,', 'amazon'], but '{dataset}' is given")

    nodes_alias2id_unknown_type = {}
    for n_type in skb.node_type_lst():
        for n_alias, node_ids in nodes_alias2id[n_type].items():
            if isinstance(node_ids, list):
                for node_id in node_ids:
                    add_node_to_node_dict(nodes_alias2id_unknown_type, n_alias, node_id)
            else:
                add_node_to_node_dict(nodes_alias2id_unknown_type, n_alias, node_ids)
    return nodes_alias2id, nodes_alias2id_unknown_type


def get_full_neighborhood(skb: SKB, start_nodes: list[int], depth: int, remove_start_nodes: bool = False):
    # Convert start_nodes to a tensor
    start_nodes = torch.tensor(start_nodes, dtype=torch.long, device=skb.sparse_adj.device)

    # Create a one-hot row vector for gt nodes:
    num_nodes = skb.sparse_adj.size(0)
    start_nodes_vec = torch.zeros(num_nodes, dtype=skb.sparse_adj.dtype, device=skb.sparse_adj.device)
    start_nodes_vec[start_nodes] = 1

    # Get depth-hop neighbors using sparse matrix multiplication:
    last_vec = start_nodes_vec
    for i in range(depth):
        last_vec = torch.sparse.mm(last_vec.unsqueeze(0), skb.sparse_adj).squeeze()

        # Remove start nodes:
    if remove_start_nodes:
        last_vec[start_nodes] = 0

        # Extract indices:
    indices = (last_vec > 0).nonzero(as_tuple=False).squeeze(1)
    return indices.tolist()



class SKBbridge:
    def __init__(self, settings: Settings, data_split: str, llm_bridge: LlmBridge,
                 skb: SKB, emb_dir: Path = None, load_embs_w_rels=True, load_embs_wo_rels=True):
        name = settings.dataset_name
        if name not in ['prime', 'mag', 'amazon']:
            raise ValueError(f"Dataset {name} not found. It should be in ['prime', 'mag,', 'amazon']")

        self.settings = settings
        self.skb = skb
        self.llm_bridge = llm_bridge

        self.nodes_alias2id, self.nodes_alias2id_unknown_type = create_node_dict(self.skb, name)

        if name == 'prime' or name == 'mag' or name == 'amazon':
            self.is_directed = False
        else:
            self.is_directed = False

        self.node_ids_by_type = {}
        for n_type in skb.node_type_lst():
            self.node_ids_by_type[n_type] = skb.get_node_ids_by_type(n_type)

        if not load_embs_w_rels and not load_embs_wo_rels:
            self.vss = None
        else:
            self.vss = VSS(skb, emb_dir, data_split, settings.get("emb_model"), settings.get("emb_model_api_url"),
                           self.node_ids_by_type, load_embs_w_rels, load_embs_wo_rels)

    def expected_answers(self, answer_ids: list[int], separator: str = ", ") -> str:
        out = ""
        if self.settings.dataset_name == 'prime':
            out += separator.join([self.skb[aid].name for aid in answer_ids])
        elif self.settings.dataset_name == 'mag':
            for aid in answer_ids:
                if "title" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['title'] + separator
                elif "DisplayName" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['DisplayName'] + separator
                else:
                    out += f"Answer has no name (!): {self.skb.node_info[aid]}{separator}"
        elif self.settings.dataset_name == 'amazon':
            for aid in answer_ids:
                if "title" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['title'] + separator
                elif "brand_name" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['brand_name'] + separator
                elif "category_name" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['category_name'] + separator
                elif "color_name" in self.skb.node_info[aid]:
                    out += self.skb.node_info[aid]['color_name'] + separator
                else:
                    out += f"Answer has no name (!): {self.skb.node_info[aid]}{separator}"
        else:
            raise NotImplementedError("unknown dataset")
        return out

    def find_closest_nodes_w_cutoff(self, target_name: str, target_type: str = None, logger: Logger = None,
                                    enable_vss: bool = False, cutoff_vss: float = None, l_max: int = 0,
                                    emb_incl_rels=True) -> list[int]:
        """
        Find node IDs whose aliases match or are similar to `target_name`, optionally augmented with vector similarity search.

        The method:
        1. First checks for exact (case-insensitive) key matches in the node alias dictionary.
        2. If fewer matches are found than `l_max` and `enable_vss=True`, it uses vector similarity search (VSS)
           to find additional nodes, filtering out already-found exact matches.
        3. Returns at most `l_max` node IDs, ordered as: exact matches first, then VSS results (in descending similarity order).

        Parameters
        ----------
        target_name : str
            The alias string to search for (case-insensitive).
        target_type : str, optional
            The node type to restrict the search to. If None, searches across all node types.
        logger : logging.Logger, optional
            If provided, logs progress messages. Otherwise, prints to stdout.
        enable_vss : bool, default False
            If True, uses vector similarity search to find additional nodes beyond exact matches.
        cutoff_vss : float, optional
            Similarity threshold (0–1) for VSS. Only nodes with score ≥ cutoff are returned.
            Required if `enable_vss=True` and `l_max=0`.
        l_max : int, default 0
            Maximum number of node IDs to return.
            If 0, returns all exact matches + VSS results (up to the total number of nodes in the dictionary).
        emb_incl_rels : bool, default True
            Whether to include relational context in the node embeddings during VSS.

        Returns
        -------
        list[int]
            List of node IDs, ordered as:
            - All exact matches (in order of appearance in the dictionary).
            - Additional nodes from VSS (in descending similarity score order, excluding exact matches).
            Truncated to at most `l_max` elements.

        Raises
        ------
        ValueError
            If `enable_vss=True`, `cutoff_vss=None`, and `l_max=0` — this combination is invalid because
            no limit or cutoff is defined for VSS.

        Notes
        -----
        - Exact matches are looked up in `self.nodes_alias2id[target_type]` (if `target_type` is given)
          or `self.nodes_alias2id_unknown_type` (if `target_type` is None).
        - VSS is performed via `self.vss.get_top_k_nodes(...)`, which returns nodes and scores sorted by similarity.
        - The final list is not re-sorted — exact matches preserve insertion order, VSS results preserve similarity order.
        - If `l_max=0`, the method returns all exact matches + all VSS results (filtered for duplicates), up to the total
          number of nodes in the relevant dictionary.

        Example
        -------
        >>> node_ids = obj.find_closest_nodes_w_cutoff("John Doe", target_type="author", enable_vss=True, cutoff_vss=0.8, l_max=5)
        >>> print(node_ids)
        [1234, 5678, 9012]  # exact match + top VSS matches (truncated to l_max=5 if more were found)
        """
        if target_type is None:
            node_dict = self.nodes_alias2id_unknown_type
        else:
            node_dict = self.nodes_alias2id[target_type]

        nodes_found = []
        nodes_direct_match = []
        num_found_key_match = 0

        if l_max == 0:
            if enable_vss and cutoff_vss is None:
                raise ValueError(
                    "Invalid combination of l_max=0, enable_vss=True and cutoff_vss=None.")
            l_max = len(node_dict)

        # search for direct key matches in node list
        if target_name.lower() in node_dict:
            node_ids = node_dict[target_name.lower()]
            if isinstance(node_ids, list):
                nodes_direct_match = node_ids
            else:
                nodes_direct_match = [node_ids]
            nodes_found = nodes_direct_match
        num_found_key_match += len(nodes_found)
        if logger is not None:
            logger.log(f"Nodes with matching alias for {target_name.lower()} directly found in database: {nodes_found}.")
        else:
            print(f"Nodes with matching alias for {target_name.lower()} directly found in database: {nodes_found}.")

        # VSS, if it is enabled. And if there are not enough direct matches found already or llm_activation is enabled
        if enable_vss and len(nodes_found) < l_max:
            node_types_to_consider = self.settings.get("nodes_to_consider")
            vss_nodes_found, vss_scores = self.vss.get_top_k_nodes(search_str=target_name, k=l_max, node_type=target_type,
                                                                   node_id_mask=None, cutoff=cutoff_vss,
                                                                   node_types_to_consider=node_types_to_consider,
                                                                   query_id=None, emb_incl_rels=emb_incl_rels)
            nodes_found = [x for x in vss_nodes_found if x not in nodes_direct_match]
            nodes_found = nodes_direct_match + nodes_found
            nodes_found = nodes_found[:l_max]

        return nodes_found

    def entity_ids2name(self, ids: list[int] | set[int], n=float("inf")) -> str:
        n = int(min(n, len(ids)))

        if not isinstance(ids, list):
            ids = list(ids)
        out = ", ".join([self.entity_id2name(idx) for idx in ids[:n]])
        if len(ids) > n:
            out += ", ..."
        return out

    def entity_id2name(self, idx: int):
        if self.settings.dataset_name == 'prime':
            return self.skb.node_info[idx]['name']
        if self.settings.dataset_name == 'mag':
            node = self.skb.node_info[idx]
            if 'title' in node:
                return node['title']
            elif 'DisplayName' in node and node['DisplayName'] != -1 and node['DisplayName'] != "-1":
                return node['DisplayName']
            else:
                return f"node without name. id: {idx}"
        elif self.settings.dataset_name == 'amazon':
            node = self.skb.node_info[idx]
            if 'title' in node:
                return node['title']
            elif 'brand_name' in node:
                return node['brand_name']
            elif "category_name" in node:
                return node['category_name']
            elif "color_name" in node:
                return node['color_name']

        raise NotImplementedError(f"Not implemented for dataset {self.settings.dataset_name}")

    def nodes2str(self, node_ids: int | list[str]) -> str:
        if isinstance(node_ids, list) or isinstance(node_ids, set):
            out = []
            for node_id in node_ids:
                out.append(self.skb.get_doc_info(node_id, add_rel=False, compact=False))
            return out
        else:
            return self.skb.get_doc_info(node_ids, add_rel=False, compact=False)

    def ground_triplets(self, triplets: list[Triplet], atomics: dict[str, TripletEnd], logger: Logger,
                        target_variable: TripletEnd, ignore_edge_labels: bool, ignore_node_labels: bool,
                        variables_in_use: set[TripletEnd]) -> {}:
        skb = self.skb

        earlier_sum_of_all_candidates = -1
        new_sum_of_all_candidates = 0

        while new_sum_of_all_candidates != earlier_sum_of_all_candidates:
            earlier_sum_of_all_candidates = new_sum_of_all_candidates
            new_sum_of_all_candidates = 0
            for triplet in triplets:
                h = triplet.h
                if ignore_edge_labels:
                    e = "*"
                else:
                    e = triplet.e

                t = triplet.t

                if h.candidates is None and t.candidates is None:
                    continue

                if h not in variables_in_use or t not in variables_in_use:
                    continue

                num_h_candidates = len(h.candidates) if h.candidates is not None else skb.num_nodes()
                num_t_candidates = len(t.candidates) if t.candidates is not None else skb.num_nodes()

                if num_h_candidates < 10000:
                    neighbors = []
                    for h_candidate in h.candidates:
                        neighbors.extend(skb.get_neighbor_nodes(h_candidate, e))
                    neighbors = set(neighbors)
                    if t.candidates is None:
                        t.candidates = neighbors
                        if t.node_type in self.node_ids_by_type and not ignore_node_labels:
                            t.intersection_update(self.node_ids_by_type[t.node_type])
                    else:
                        t.intersection_update(neighbors)

                    logger.log(
                        f"Found {len(t.candidates)} candidates for tail {t.get_uid()} ({t.node_type}).")
                else:
                    logger.log(f"{triplet=}: Too many candidates ({num_h_candidates}) for triplet head to search "
                               f"for all their neighbors.")
                if num_t_candidates < 10000:
                    neighbors = []
                    for t_candidate in t.candidates:
                        neighbors.extend(skb.get_neighbor_nodes(t_candidate, e))
                    neighbors = set(neighbors)
                    if h.candidates is None:
                        h.candidates = neighbors
                        if h.node_type in self.node_ids_by_type and not ignore_node_labels:
                            h.intersection_update(self.node_ids_by_type[h.node_type])
                    else:
                        h.intersection_update(neighbors)

                    logger.log(
                        f"Found {len(h.candidates)} candidates for head {h.get_uid()} ({h.node_type}).")
                else:
                    logger.log(f"{triplet=}: Too many candidates ({num_t_candidates}) for triplet tail to search "
                               f"for all their neighbors.")
                num_h_candidates_new = len(h.candidates) if h.candidates is not None else skb.num_nodes()
                num_t_candidates_new = len(t.candidates) if t.candidates is not None else skb.num_nodes()
                logger.log(
                    f"Grounded triplet: {h.get_uid()} ({h.node_type}) [{num_h_candidates_new}/{num_h_candidates} candidates]-> "
                    f"{e} -> {t.get_uid()} ({t.node_type}) [{num_t_candidates_new}/{num_t_candidates} candidates].")
            for atomic in atomics.values():
                if atomic.candidates is not None:
                    new_sum_of_all_candidates += len(atomic.candidates)
            if target_variable.candidates is not None and len(target_variable.candidates) == 0:
                logger.log("No candidates for target variable.")
                break
        return atomics