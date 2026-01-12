from pathlib import Path

import pandas as pd

ROOT_PATH = Path(__file__).parent.parent.absolute() / "output"

def print_stats(dataset = "prime", split = "human_generated_eval", experiment_name = "unnamed_experiment", llm="gpt-oss-120b",
                steps = [1], cols = None):

    for step in steps:
        print(f"step {step}:")
        if step == 1:
            file_name = "step1_target_type.csv"
        elif step == 3:
            file_name = "step3_regex.csv"
        elif step == 4:
            file_name = "step4_ground_triplets.csv"
        elif step == 67:
            file_name = "step6_and_7_vss.csv" 

        dir_path = ROOT_PATH / dataset / split / llm / experiment_name
        df = pd.read_csv(dir_path / file_name, delimiter=',', quotechar='|')

        if cols is None:
            cols = df.columns

        for col_name in cols:
            try:
                df[col_name] = df[col_name].astype(float)
                print(col_name, df[col_name].sum(), f'{df[col_name].sum() / len(df):.3f}', len(df))
            except:
                pass