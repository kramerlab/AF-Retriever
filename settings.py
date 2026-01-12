import json
from pathlib import Path


class Settings:
    def __init__(self, dataset_name:str, llm_model:str=None, emb_model: str=None, configs_path: str=None):
        # load configs from file
        if configs_path is None:
            self.configs_path = Path(__file__).parent / "configs.json"
        else:
            self.configs_path = Path(configs_path)
        self.dataset_name = dataset_name

        # read file
        with open(self.configs_path, "r") as json_file:
            configs = json.load(json_file)

        print(f"Loaded configs from {self.configs_path}.")

        # load dataset specific configs
        data_specific_configs = configs[dataset_name]
        self.configs = configs["general"]
        self.configs.update(data_specific_configs)
        self.configs["llm"] = configs["llm"]

        # replace default settings (from config file) by method arguments
        if emb_model is not None:
            self.configs["emb_model"] = emb_model
        if llm_model is not None:
            self.configs["llm"]["llm_model"] = llm_model

    def get(self, config_name: str):
        return self.configs[config_name]

    def edge_type2str(self, key: str) -> str:
        return self.configs["edge_type2str"][key]
