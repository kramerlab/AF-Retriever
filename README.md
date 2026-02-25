# Autofocus-Retriever (AF-Retriever)
For a detailed description of AF-Retriever, please read our papter at [https://arxiv.org/abs/2505.09246](https://arxiv.org/abs/2505.09246)

## Getting Started


### Summary
1. Install the required Python packages: `pip install stark-qa langchain==0.0.316` (Python >=3.8 and <3.12).
2. STaRK \[0\] datasets and SKBs will be automatically downloaded on use.
3. Download the node embeddings for each SKB to the directory `emb` from: https://zenodo.org/records/17723545?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc2NDE3MTI4MSwiZXhwIjoxNzgwMTg1NTk5fQ.eyJpZCI6IjM5N2ZiY2IyLWY0YWYtNGVjNy05MTVkLWIxNGE4ZjJhZTdjYiIsImRhdGEiOnt9LCJyYW5kb20iOiJhMWViMDVjODRiZTI3NzA5M2VkMDBlOTY4MDQwNGZhZCJ9.57ZzX_0I9PavqUz2ItI5FDXdQpOJyAWmwR_Xp5hMVPeOWr6pf8ZwpW-6a6lsMYZ6MRZ_Ud_rmGr3FXzAnBihgw
4. Extract the embeddings without renaming files and folders.
5. For API-based embedding model and LLM use (recommended), paste your API key(s) in the `.env` file. 
6. Adjust the configuration settings in `configs.json`. E.g., set your API url, which is compatible with the OpenAI interface. 
7. Run `main.py` with the arguments defined in the `parse_args()` function.


### Individual Steps:
#### 1. Python Environment Setup
The stark-qa package, which allows using the linked knowledge bases, requires Python >=3.8 and <3.12. 
We recommend creating a new virtual environment:
```bash
conda  create -n AF_Retriever_env python=3.11
conda activate AF_Retriever_env
pip install --upgrade pip
```
Install pytorch for your cuda version if you have a cuda-compatible GPU. 
You can query the version with `nvidia-smi`. 
If you do not have a cuda-compatible GPU or do not want to use it, install the CPU version instead.
```bash
# Replace cu130 in the next line by your cuda version.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 
# For CPU alternative: 
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
The installation of stark-qa includes all other requirements AF-Retriever has. 
To avoid compatibility issues, specify the langchain version.
```bash
pip install stark-qa langchain==0.0.316 dotenv
```
In case of compatibility issues, try setting up a pip venv from requirements.txt.

####  2. Other prerequisites
The init function of `framework.py` loads several prerequisites: 
- The question-answer dataset, 
- the Semi-Structured Knowledge Bases (SKBs), 
- the base LLM and embedding model, 
- the node embeddings.

##### a) QA Datasets and SKBs
AF-Retriever supports SKBs of the python package `stark_qa.skb` [0] . 
The SKBs and QA sets are downloaded automatically from huggingface when used. 
If either is downloaded already, set `skb_path` to the root directory of your SKBs in `configs.json`. 
Otherwise, leave the default: "skb_path": "auto_download".

##### b) Embeddings
You can download embeddings of all vertices in the STaRK SKBs from OpenAI's model `text-embedding-3-small`
Download from https://zenodo.org/records/17723545?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc2NDE3MTI4MSwiZXhwIjoxNzgwMTg1NTk5fQ.eyJpZCI6IjM5N2ZiY2IyLWY0YWYtNGVjNy05MTVkLWIxNGE4ZjJhZTdjYiIsImRhdGEiOnt9LCJyYW5kb20iOiJhMWViMDVjODRiZTI3NzA5M2VkMDBlOTY4MDQwNGZhZCJ9.57ZzX_0I9PavqUz2ItI5FDXdQpOJyAWmwR_Xp5hMVPeOWr6pf8ZwpW-6a6lsMYZ6MRZ_Ud_rmGr3FXzAnBihgw
Extract the downloaded .zip file in a new directory "emb", such that, e.g., for the SKB prime, emb/prime/text-embedding-3-small/nodes/node_embeddings_add_rel_not_compact.pt is available.
For each SKB, there exists one file with all embeddings where relations were included in the text exmbeddings, and one file with all embeddings without.

##### c) Embedding Model and Large Language Model 
Currently, only API-based embedding model access is supported via the openai API. Adaptions can be implemented in `vss.py`.
For API-based embedding mode use, paste your (OpenAI) API key in the `.env` file.
For API-based LLM use, past your API key in the `.env` file. 
Configure the corresponding model name and API urls (for both embedding model and llm) in the configs_file (see below).
Alternatively, modify `local_llm_bridge` to suit your local LLM setup.


#### 3. configs.json
Adjust paths, hyperparameters, and other configuration settings in `configs.json`.


#### 4. Usage: Run main.py
To run and evaluate the AF-Retriever framework, you may use `main.py`. 
Required arguments are the name (currently supported: prime, mag, or amazon) and split (e.g., validation, test, human_generated_eval).
Run `python main.py --help` for a full description of available arguments.

Minimal example:
```bash
python -m main --dataset prime --split human_generated_eval
```
## External References:
[0] STaRK [https://github.com/snap-stanford/stark](https://github.com/snap-stanford/stark)




