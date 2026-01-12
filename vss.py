import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from openai import OpenAI

from stark_qa.skb import SKB
from logger import Logger


def load_emb_model(model_name: str, emb_model_url: str | None = None):
    load_dotenv()
    if (model_name == "text-embedding-ada-002" or model_name == "text-embedding-3-small"
            or model_name == "text-embedding-3-large"):
        return OpenAI(api_key=os.environ.get("EMB_MODEL_API_KEY"), base_url=emb_model_url)
    else:
        raise ValueError(f"Embeddings model {model_name} is not supported.")


class VSS:
    def __init__(self,
                 skb: SKB,
                 emb_dir: Path,
                 data_split: str,
                 emb_model_name: str,
                 emb_model_api_url: str,
                 node_ids_by_type: dict,
                 load_node_embs_w_rels: bool,
                 load_node_embs_wo_rels: bool
                 ):
        """
        Initializes the VSS + LLMReranker model.
        Loads all embeddings in access efficient tensors.
        Args:
            skb (SemiStruct): Knowledge base.
            emb_dir (Path): Path to directory with all embedding.
            data_split (str): ["train", "val", "test", "human_generated_eval"] Data split mode determining which
                query embeddings to load.
            emb_model_name (str): Embedding model name.
            node_ids_by_type (dict): Dictionary grouping list of node IDs by their node types.
            load_node_embs_w_rels: Whether to load node embeddings with relations in embedded documents.
            load_node_embs_wo_rels: Whether to load node embeddings without relations in embedded documents.
        """

        if not load_node_embs_w_rels and not load_node_embs_wo_rels:
            print("VSS offline mode.")
            return

        self.skb = skb
        self.candidate_ids = skb.candidate_ids
        self.query_emb_dict = {}
        self.emb_model_name = emb_model_name
        self.node_ids_by_type = node_ids_by_type
        self.emb_client = load_emb_model(emb_model_name, emb_model_api_url)

        # loading several embeddings:
        # 1) queries: questions from test sets (lazy loading)
        # 2) nodes: vertices in SKB
        # 3) entities: other strings representing entities that have been searched for already

        if emb_model_name == "text-embedding-ada-002-STARK":
            emb_dir /= "text-embedding-ada-002"
        else:
            emb_dir /= emb_model_name

        # 1) query embeddings
        if data_split == "human_generated_eval":
            self.query_emb_dir = emb_dir / "queries_human_generated_eval"
        else:
            self.query_emb_dir = emb_dir / "queries"
        query_emb_dict_path = self.query_emb_dir / 'query_emb_dict.pt'


        self.query_emb_dict = {}
        if query_emb_dict_path.exists():
            print(f'Loading query embeddings from {query_emb_dict_path}.')
            self.query_emb_dict = torch.load(query_emb_dict_path)

        # 2) node embeddings
        nodes_emb_dir = emb_dir / "nodes"
        if load_node_embs_w_rels:
            nodes_emb_dir.mkdir(parents=True, exist_ok=True)
            self.node_emb_path = nodes_emb_dir / 'node_embeddings_add_rel_not_compact.pt'
            if self.node_emb_path.is_file():
                ungrouped_emb_dict = torch.load(self.node_emb_path)
            else:
                raise FileNotFoundError(f"Entity embeddings not found at {self.node_emb_path}. "
                                        f"Make sure the correct path is set or start main of "
                                        f"additional_code/create_embeddings.py to regenerate them.")
            print(f'Loaded {len(ungrouped_emb_dict["indices"])} node embeddings from {self.node_emb_path}!')

            self.node_emb_dict = {}
            for node_type in skb.node_type_lst():
                mask = [x == node_type for x in ungrouped_emb_dict["node_labels"]]
                self.node_emb_dict[node_type] = {
                    "indices": torch.IntTensor(ungrouped_emb_dict["indices"])[mask].tolist(),
                    "embeddings": ungrouped_emb_dict["embeddings"][mask, :]
                }

            for node_type in skb.node_type_lst():
                assert len(self.node_emb_dict[node_type]["indices"]) == len(self.node_ids_by_type[node_type]), \
                    (f"number of node embeddings ({len(self.node_emb_dict[node_type]['indices'])}) does not match number "
                     f"of nodes in the SKB ({len(self.node_ids_by_type[node_type])}). {node_type=}.")


        # Node embeddings with relations not included in embeddings
        if load_node_embs_wo_rels:
            if emb_model_name != "text-embedding-ada-002-STARK":
                self.node_emb_no_rel_path = nodes_emb_dir / 'node_embeddings_no_rel_not_compact.pt'
                if self.node_emb_no_rel_path.exists():
                    ungrouped_emb_dict = torch.load(self.node_emb_no_rel_path)
                else:
                    raise FileNotFoundError(f"Entity embeddings not found at {self.node_emb_no_rel_path}."
                                            f"Make sure the correct path is set or start main of "
                                            f"additional_code/create_embeddings.py to regenerate them.")
                print(f'Loaded {len(ungrouped_emb_dict["indices"])} node embeddings from {self.node_emb_no_rel_path}!')

                self.node_emb_no_rel_dict = {}
                for node_type in skb.node_type_lst():
                    mask = [x == node_type for x in ungrouped_emb_dict["node_labels"]]
                    self.node_emb_no_rel_dict[node_type] = {
                        "indices": torch.IntTensor(ungrouped_emb_dict["indices"])[mask].tolist(),
                        "embeddings": ungrouped_emb_dict["embeddings"][mask, :]
                    }
        if load_node_embs_w_rels and load_node_embs_wo_rels:
            for node_type in skb.node_type_lst():
                assert len(self.node_emb_no_rel_dict[node_type]["indices"]) == len(self.node_ids_by_type[node_type]), \
                    (f"number of node embeddings ({len(self.node_emb_no_rel_dict[node_type]['indices'])}) does not match number "
                     f"of nodes in the SKB ({len(self.node_ids_by_type[node_type])}). {node_type=}.")


        # 3) 'open' string embeddings
        entities_emb_dir = emb_dir / "entities"
        entities_emb_dir.mkdir(parents=True, exist_ok=True)
        self.entity_emb_path = entities_emb_dir / 'entity_emb_dict.pt'
        if self.entity_emb_path.exists():
            self.entity_emb_dict = torch.load(self.entity_emb_path)
        else:
            self.entity_emb_dict = {}
        print(f'Loaded {len(self.entity_emb_dict)} entity embeddings from {self.entity_emb_path}!')



    def get_embedding(self, query: str, model: str):
        emb = self.emb_client.embeddings.create(input=query, model=model)
        return torch.FloatTensor(emb.data[0].embedding)

    def compute_similarities(self,
                             query_emb: torch.Tensor,
                             node_type: str,
                             node_id_mask: list[int] | set[int],
                             node_ids_to_exclude: list[int] | set[int] = [],
                             emb_incl_rels: bool = True) -> dict:
        """
        Forward pass to compute similarity scores for the given query.

        Args:
            query_emb (torch.Tensor): Query embedding.
            emb_incl_rels: True if relations should be included in embeddings, else False.
            node_type: Type of nodes to be returned.
            node_id_mask: A list or set of node IDs to be considered.
            node_ids_to_exclude: A list or set of node IDs to be NOT considered.

        Returns:
            pred_dict (dict): A dictionary of node ids and their corresponding similarity scores.
        """
        if emb_incl_rels:
            emb_dict = self.node_emb_dict
        else:
            emb_dict = self.node_emb_no_rel_dict
        similarity = torch.matmul(emb_dict[node_type]["embeddings"], query_emb)

        node_ids = emb_dict[node_type]["indices"]
        score_dict = {node_ids[i]: similarity[i] for i in range(len(similarity))}

        # filter score dict by masks
        if node_id_mask is not None:
            filtered_score_dict = {}
            for node_id in node_id_mask:
                if node_id in score_dict:
                    filtered_score_dict[node_id] = score_dict[node_id]
            score_dict = filtered_score_dict
        if len(node_ids_to_exclude) > 0:
            for node_id in node_ids_to_exclude:
                if node_id in score_dict.keys():
                    score_dict.pop(node_id)
        return score_dict

    def get_query_emb(self,
                      query: str,
                      query_id: int,
                      emb_model: str = None) -> torch.Tensor:
        """
        Retrieves or computes the embedding for the given query or entity.

        Args:
            query (str): Query string.
            query_id (int): Query index.
            emb_model (str): Embedding model to use.

        Returns:
            query_emb (torch.Tensor): Query embedding.
        """
        if emb_model is None:
            emb_model = self.emb_model_name

        # loading embedding of free text (entity embedding) not question embedding from dataset:
        if query_id is None:
            # load embedding from cache if available:
            if query in self.entity_emb_dict:
                query_emb = self.entity_emb_dict[query]
            # retrieve embedding if it is not in the cache:
            else:
                query_emb = self.get_embedding(query, model=emb_model)
                self.entity_emb_dict[query] = query_emb
                torch.save(self.entity_emb_dict, self.entity_emb_path)
                print(f'Entity embedding for "{query}" saved to {self.entity_emb_path}.')

        # return preloaded query embedding
        elif query_id in self.query_emb_dict:
            query_emb = self.query_emb_dict[query_id]
        else:
            # load single query embedding from single file
            if not self.query_emb_dir.exists():
                self.query_emb_dir.mkdir(parents=True, exist_ok=True)
            query_emb_dict_path = self.query_emb_dir / f'query_{query_id}.pt'
            if query_emb_dict_path.exists():
                query_emb = torch.load(query_emb_dict_path).reshape(-1)
                #print(f'Query embedding loaded from {query_emb_dict_path}')
            else:
                query_emb = self.get_embedding(query, model=emb_model)
                torch.save(query_emb, query_emb_dict_path)
                print(f'Query embedding saved to {query_emb_dict_path}')
        return query_emb

    def get_top_k_nodes(self, search_str: str, k: int, node_type: str | None, logger: Logger = None,
                        node_id_mask: list[int] | set[int] = None, complement_with_non_masked_ids=False,
                        query_id: int = None, node_ids_to_exclude: list[int] | set[int] = [],
                        node_types_to_consider: list[str] = [], cutoff: float=0.0, emb_incl_rels: bool = True) -> tuple[list, list]:
        """
        Searches for the k nodes with the highest cosine similarity to a search string.
        Args:
            search_str (str): Search string
            k (int): Number of nodes to return
            node_type (str | None): Type of nodes to return. None to return all nodes.
            logger (Logger): optional
            node_id_mask (list[int] | set[int] = None): List or set of nodes to prefer (will be returned first)
            complement_with_non_masked_ids (bool) : Whether to include more nodes if the sum of elements in node_id_mask
                is smaller than k.
            query_id (int): ID of a query in the dataset. If it is not None, it replaces the search string
                answer the query.
            node_ids_to_exclude (list[int] | set[int] = []]): List or set of node IDs to exclude.
            node_types_to_consider (list[str]): List of available node types if node_id_mask is None.
            cutoff (float): Boundary for cosine similarity. All nodes with similarity below cutoff will be discarded. Default is 0.0.
            emb_incl_rels: Set True if entity relations should be included in embeddings, else False.

        Returns:
            list: the k closest nodes
        """
        if cutoff is None:
            cutoff = 0.0
        query_emb = self.get_query_emb(search_str, query_id)
        if node_type is None:
            score_dict = {}
            for n_type in node_types_to_consider:
                score_dict.update(
                    self.compute_similarities(query_emb=query_emb, node_type=n_type,node_id_mask=node_id_mask,
                                              node_ids_to_exclude=node_ids_to_exclude, emb_incl_rels=emb_incl_rels))
        else:
            score_dict = self.compute_similarities(query_emb=query_emb, node_type=node_type,node_id_mask=node_id_mask,
                                                   node_ids_to_exclude=node_ids_to_exclude, emb_incl_rels=emb_incl_rels)

        # Remove nodes whose similarity is below cutoff boundary
        if cutoff > 0.0:
            filtered_dict = {}
            for key in score_dict:
                if score_dict[key] >= cutoff:
                    filtered_dict[key] = score_dict[key]
            score_dict = filtered_dict

        # Get top k node IDs based on their similarity
        node_scores = list(score_dict.values())
        top_k_idx = torch.topk(torch.FloatTensor(node_scores), min(k, len(node_scores)), dim=-1, largest=True,
                               sorted=True).indices.tolist()
        # Convert score_dict.keys() to a tensor for efficient indexing
        vss_scores = torch.tensor(node_scores)[top_k_idx].tolist()
        keys_tensor = torch.tensor(list(score_dict.keys()), dtype=torch.long)

        # Use advanced indexing to get the top-k node IDs
        top_k_node_ids = keys_tensor[top_k_idx].tolist()

        # for target variable
        if complement_with_non_masked_ids and node_id_mask is not None and len(node_id_mask) < k:
            additional_node_ids, additional_vss_scores = self.get_top_k_nodes(search_str, k - len(node_id_mask), node_type,
                                                                              logger=logger,
                                                       complement_with_non_masked_ids=False, query_id=query_id,
                                                       node_ids_to_exclude=top_k_node_ids,
                                                       cutoff=cutoff,
                                                       node_types_to_consider=node_types_to_consider,
                                                                              emb_incl_rels=emb_incl_rels)

            top_k_node_ids += additional_node_ids
            vss_scores += additional_vss_scores
            if logger is not None:
                logger.log(f"VSS: Added further answers to candidate list. New list (top10): {top_k_node_ids[:10]}")
            else:
                print(f"VSS: Added further answers to candidate list. New list (top10): {top_k_node_ids[:10]}")

        return top_k_node_ids, vss_scores
