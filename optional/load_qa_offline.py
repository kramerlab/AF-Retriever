import os
from typing import Union
import pandas as pd
from stark_qa.retrieval import STaRKDataset

REGISTERED_DATASETS = [
    'amazon',
    'prime',
    'mag'
]

def load_qa_offline(name: str,
                    root: Union[str, None] = None,
                    human_generated_eval: bool = False) -> STaRKDataset:
    """
    Load the QA dataset.

    Args:
        name (str): Name of the dataset. One of 'amazon', 'prime', or 'mag'.
        root (Union[str, None]): Root directory to store the dataset. If not provided, the default Hugging Face cache path is used.
        human_generated_eval (bool): Whether to use human-generated evaluation data. Default is False.

    Returns:
        STaRKDataset: The loaded STaRK dataset.

    Raises:
        ValueError: If the dataset name is not registered.
    """
    assert name in REGISTERED_DATASETS, f"Unknown dataset {name}"

    if root is not None:
        if not os.path.isabs(root):
            root = os.path.abspath(root)


    """
    Initialize the STaRK dataset.

    Args:
        name (str): Name of the dataset.
        root (Union[str, None]): Root directory to store the dataset. If None, default HF cache paths will be used.
        human_generated_eval (bool): Whether to use human-generated evaluation data.
    """
    stark_dataset = object.__new__(STaRKDataset)
    stark_dataset.name = name
    stark_dataset.root = root
    stark_dataset.dataset_root = os.path.join(stark_dataset.root, name) if stark_dataset.root is not None else None
    # stark_dataset._download()
    stark_dataset.split_dir = os.path.join(stark_dataset.dataset_root, 'split')
    stark_dataset.query_dir = os.path.join(stark_dataset.dataset_root, 'stark_qa')
    stark_dataset.human_generated_eval = human_generated_eval

    stark_dataset.qa_csv_path = os.path.join(
        stark_dataset.query_dir,
        'stark_qa_human_generated_eval.csv' if human_generated_eval else 'stark_qa.csv'
    )

    stark_dataset.data = pd.read_csv(stark_dataset.qa_csv_path)
    stark_dataset.indices = sorted(stark_dataset.data['id'].tolist())
    stark_dataset.split_indices = stark_dataset.get_idx_split()

    return stark_dataset


