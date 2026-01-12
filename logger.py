import ast
import csv
import sys
from pathlib import Path

import torch
import triplet

torch.serialization.add_safe_globals([triplet.TripletEnd])

csv.field_size_limit(sys.maxsize)

from triplet import TripletEnd, Triplet


def calculate_metrics(answer_ids, ground_truth_ids) -> dict[str, float]:
    hit_1, hit_5, hit_20, hit_40, hit_50 = 0, 0, 0, 0, 0
    reciprocal_rank, reciprocal_rank_20 = 0.0, 0.0
    hits = 0
    for i in range(len(answer_ids)):
        if answer_ids[i] in ground_truth_ids:
            if i < 1:
                hit_1 = 1
            if i < 5:
                hit_5 = 1
            if i < 20:
                hits += 1
                hit_20 = 1
            if i < 40:
                hit_40 = 1
            if i < 50:
                hit_50 = 1
            if reciprocal_rank <= 0.0:
                reciprocal_rank = 1.0 / (i + 1)
            if reciprocal_rank_20 <= 0.0 and i < 20:
                reciprocal_rank_20 = 1.0 / (i + 1)
    recall_20 = hit_20 / min(20, len(ground_truth_ids))
    return {
        "hit_1": hit_1,
        "hit_5": hit_5,
        "hit_20": hit_20,
        "hit_40": hit_40,
        "hit_50": hit_50,
        "reciprocal_rank": reciprocal_rank,
        "reciprocal_rank_20": reciprocal_rank_20,
        "recall_20": recall_20,
    }


class Step1PredictTargetTypeResult:
    def __init__(self, target_type: str, is_invalid: bool, is_incorrect: bool,
                 ground_truth: str):
        self.target_type = target_type
        if self.target_type == "":
            self.target_type = None
        self.is_invalid = is_invalid
        self.is_incorrect = is_incorrect
        self.ground_truth = ground_truth


class Step2DeriveCypherResult:
    def __init__(self, cypher_str: str = None):
        self.cypher_str = cypher_str

    def save_to_file(self, question_id: int, query: str, output_path: str | Path):
        file_path = Path(output_path) / "step2_derive_cypher.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,query,cypher_str\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, query, self.cypher_str])


class Step3RegexResult:
    def __init__(self, target_type: str, triplets: list[Triplet], symbols: dict[str, TripletEnd],
                 target_variable: TripletEnd, error_message: str, target_var_not_in_triplets: bool):
        self.target_type = target_type
        self.triplets = triplets
        self.symbols = symbols
        self.target_variable = target_variable
        self.error_message = error_message
        self.target_var_not_in_triplets = target_var_not_in_triplets

        self.target_type_is_invalid = None
        self.target_type_is_incorrect = None
        self.target_type_ground_truth = None

        self.num_valid_constants = None
        self.num_valid_variables = None

    def set_target_type_pred(self, target_type: str, is_invalid: bool, is_incorrect: bool,
                             ground_truth: str):
        self.target_type = target_type
        self.target_type_is_invalid = is_invalid
        self.target_type_is_incorrect = is_incorrect
        self.target_type_ground_truth = ground_truth

    def save_to_file(self, question_id: int, query: str, output_path: str | Path):
        symbols = symbols_to_str("ERROR:" in self.error_message, self.symbols, only_uids=False)
        triplets = triplets_to_str(self.triplets)
        target_variable = "" if self.target_variable is None else self.target_variable.get_uid()
        try:
            num_triplets = len(self.triplets)
        except TypeError:
            num_triplets = 0

        # save to file
        file_path = Path(output_path) / "step3_regex.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,query,target_type,tt_ground_truth,tt_invalid,tt_incorrect,symbols,triplets,"
                                  "target_variable,errors,target_var_not_in_triplets,num_constants,num_variables,num_triplets\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, query, self.target_type, self.target_type_ground_truth,
                             int(self.target_type_is_invalid), int(self.target_type_is_incorrect),
                             symbols, triplets, target_variable, self.error_message,
                             int(self.target_var_not_in_triplets),
                             self.num_valid_constants, self.num_valid_variables, num_triplets])


class Step4SymbolCandidatesResult:
    def __init__(self, skipped: bool = False):
        self.valid_symbols : dict[str, TripletEnd] | None = None
        self.skipped = skipped

    def save_to_file(self, question_id: int, query: str, output_path: str | Path):
        constants = symbols_to_str(self.skipped, self.valid_symbols)

        # save to file
        file_path = Path(output_path) / "step4_symbol_candidates.csv"
        if not file_path.exists():
            with open(file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,query,constant_candidates,skipped\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, query, constants, int(self.skipped)])

        (Path(output_path) / "step4_constants").mkdir(parents=True, exist_ok=True)
        torch.save(self.valid_symbols, Path(output_path) / "step4_constants" / f"s4c_{question_id}.pt")

class Step5GroundTripletsResult:
    def __init__(self, answers: list[set[int]], answers_flattened: list[int],
                 num_variables_wo_candidates: int, num_variable_candidates: int, l_first_hit: int, l_last_hit: int,
                 skipped: bool = False):
        self.answers = answers
        self.answers_flattened = answers_flattened
        self.num_variable_candidates = num_variable_candidates
        self.num_variables_wo_candidates = num_variables_wo_candidates
        self.l_first_hit = l_first_hit
        self.l_last_hit = l_last_hit
        self.skipped = skipped
        self.num_true_pos_in_prefilter = 0
        self.num_false_pos_in_prefilter = 0
        self.precision = 0
        self.recall = 0

    def save_to_file(self, question_id: int, output_path: str | Path, ground_truths: list[int]):
        file_path = Path(output_path) / "step5_ground_triplets.csv"
        metrics = calculate_metrics(self.answers_flattened, ground_truths)

        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,answers,answers_flattened,variables_without_candidates,variable_candidates,"
                                  "target_candidates,true_pos,false_pos,precision,recall,l_first_hit,l_last_hit,"
                                  "hit_1,hit_5,hit_20,hit_40,hit_50,recall_20,reci_rank,reci_rank_20,skipped\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, self.answers, self.answers_flattened, self.num_variables_wo_candidates,
                             self.num_variable_candidates, len(self.answers_flattened), self.num_true_pos_in_prefilter,
                             self.num_false_pos_in_prefilter, self.precision, self.recall, self.l_first_hit,
                             self.l_last_hit,
                             metrics["hit_1"], metrics["hit_5"], metrics["hit_20"], metrics["hit_40"],
                             metrics["hit_50"], metrics["recall_20"], metrics["reciprocal_rank"],
                             metrics["reciprocal_rank_20"], int(self.skipped)])


class Step6VSSResult:
    def __init__(self, vss_top_hits: list[int], vss_scores: list, fallback_solution: bool = False):
        self.vss_top_hits = vss_top_hits
        self.fallback_solution = fallback_solution
        self.ground_truths = None
        self.vss_scores = vss_scores


class Step7VSSResult:
    def __init__(self, vss_top_hits: list[int], vss_scores: list, fallback_solution: bool = False):
        self.vss_top_hits = vss_top_hits
        self.fallback_solution = fallback_solution
        self.ground_truths = None
        self.vss_scores = vss_scores


class Step6plus7VSSResult:
    def __init__(self, step6_result: Step6VSSResult, step7_result: Step7VSSResult, alpha:int, k: int):
        if k == 0:
            return
        answers_combined = step6_result.vss_top_hits[:alpha]
        vss_scores_combined = step6_result.vss_scores[:alpha]
        j = 0

        while len(answers_combined) < k:
            if step7_result.vss_top_hits[j] not in answers_combined:
                answers_combined.append(step7_result.vss_top_hits[j])
                vss_scores_combined.append(step7_result.vss_scores[j])
            j += 1
        self.vss_top_hits = answers_combined
        self.vss_scores = vss_scores_combined
        self.ground_truths = None

        self.answer_sources = []
        for i in range(k):
            if self.vss_top_hits[i] in step6_result.vss_top_hits[:k]:
                if self.vss_top_hits[i] in step7_result.vss_top_hits[:k]:
                    self.answer_sources.append("both")
                else:
                    self.answer_sources.append("constrained-based")
            else:
                self.answer_sources.append("vss-based")


class Step8FinalRerankerResult:
    def __init__(self, answer_ids: list[int], answer_str: str, fallback_solution: bool = False):
        self.final_answer_str = answer_str
        self.answer_ids = answer_ids
        self.fallback_solution = fallback_solution
        self.ground_truth_str = None
        self.ground_truths = None


def load_symbols_from_str(skipped: bool, symbols_str: str, only_uids: bool = False) -> dict[str, TripletEnd]:
    if skipped or symbols_str == "":
        return None
    symbols = {}
    for symbol in symbols_str.split("<<<>>>"):
        uid = symbol.split(", self.is_constant=")[0]
        name, node_type = uid.split("::")
        if node_type == "None":
            node_type = None
        if only_uids:
            is_constant = True
            properties = None
        else:
            is_constant = symbol.split(", self.is_constant=")[1]

            is_constant = is_constant.split(", self.properties=")[0]
            is_constant = is_constant == "True"

            properties = symbol.split(", self.properties=")[1].split(", num_candidates=")[0]
            properties = ast.literal_eval(properties)
        triplet_end = TripletEnd(name, node_type, is_constant, candidates=None)
        if not only_uids:
            triplet_end.properties = properties
        symbols[triplet_end.get_uid()] = triplet_end
    return symbols


def symbols_to_str(skipped: bool, symbols: dict[str, TripletEnd], only_uids: bool = False) -> str:
    if skipped:
        return ""
    elif only_uids:
        return "<<<>>>".join([x.get_uid() for x in list(symbols.values())])
    else:
        return "<<<>>>".join([str(x) for x in list(symbols.values())])


def triplets_to_str(triplets: list[Triplet]):
    if triplets is None or len(triplets) == 0:
        return ''
    else:
        return "<<<>>>".join([str(t) for t in triplets])


def str_to_triplets(skipped: bool, triplet_str: str, symbols: dict[str, TripletEnd]):
    triplets = []
    if not skipped and triplet_str != "":
        triplets_str = triplet_str.split("<<<>>>")

        for triplet in triplets_str:
            triplet = triplet.split(" -> ")
            try:
                h = symbols[triplet[0]]
                e = triplet[1]
                r = symbols[triplet[2]]
                triplets.append(Triplet(h, e, r))
            except KeyError:
                print("STOP")
    return triplets


class Logger:
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.qa_stats = {}

        self.step1_results = {}
        self.step2_results = {}
        self.step3_results = {}
        self.step4_results = {}
        self.step5_results = {}
        self.step6_results = {}
        self.step7_results = {}
        self.step6_plus_7_results = {}
        self.step8_results = {}

    def get_step1_result(self, question_id) -> Step1PredictTargetTypeResult:
        return self.step1_results[question_id]

    def get_step2_result(self, question_id) -> Step2DeriveCypherResult:
        return self.step2_results[question_id]

    def get_step3_result(self, question_id) -> Step3RegexResult:
        return self.step3_results[question_id]

    def get_step4_result(self, question_id, overwrite_path) -> Step4SymbolCandidatesResult:
        r = self.step4_results[question_id]
        if r.valid_symbols is None:
            if overwrite_path is None:
                file_path = self.output_path / "step4_constants" / f"s4c_{question_id}.pt"
            else:
                file_path = Path(overwrite_path) / "step4_constants" / f"s4c_{question_id}.pt"

            r.valid_symbols = dict(torch.load(file_path, map_location=torch.device('cpu')))
        return r

    def get_step5_result(self, question_id) -> Step5GroundTripletsResult:
        return self.step5_results[question_id]

    def get_step6_result(self, question_id) -> Step6VSSResult:
        return self.step6_results[question_id]

    def get_step7_result(self, question_id) -> Step7VSSResult:
        return self.step7_results[question_id]

    def get_step6_plus_7_result(self, question_id) -> Step6plus7VSSResult:
        return self.step6_plus_7_results[question_id]

    def get_step8_result(self, question_id) -> Step8FinalRerankerResult:
        return self.step8_results[question_id]

    def log(self, text: str, print_to_console: bool = True):
        if print_to_console:
            print(text)

        # Open the log_file file in append mode ('a')
        with (open(self.output_path / "log.txt", 'a', encoding='utf-8', errors='replace') as log_file):
            log_file.write(text + "\n")


    def save_step1(self, question_id: int, r: Step1PredictTargetTypeResult):
        self.step1_results[question_id] = r

        # save to file
        file_path = self.output_path / "step1_target_type.csv"
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,target_type,ground_truth,invalid,incorrect\n")
        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, r.target_type, r.ground_truth, int(r.is_invalid), int(r.is_incorrect)])

    def load_step1(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step1_target_type.csv"
        else:
            file_path = Path(overwrite_path) / "step1_target_type.csv"

        if file_path.exists():
            self.log("Step1 results loaded.")
        else:
            self.log("Step1 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step1PredictTargetTypeResult(row['target_type'], bool(int(row["invalid"])), bool(int(row["incorrect"])),
                                                 row["ground_truth"])
                self.step1_results[int(row["q_id"])] = r

    def save_step2(self, question_id: int, query: str, r: Step2DeriveCypherResult):
        self.step2_results[question_id] = r
        r.save_to_file(question_id, query, self.output_path)

    def load_step2(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step2_derive_cypher.csv"
        else:
            file_path = Path(overwrite_path) / "step2_derive_cypher.csv"

        if file_path.exists():
            self.log("Step2 results loaded.")
        else:
            self.log("Step2 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step2DeriveCypherResult(row["cypher_str"])
                self.step2_results[int(row["q_id"])] = r

    def save_step3(self, question_id: int, query: str, r: Step3RegexResult):
        self.step3_results[question_id] = r
        r.save_to_file(question_id, query, self.output_path)

    def load_step3(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step3_regex.csv"
        else:
            file_path = Path(overwrite_path) / "step3_regex.csv"

        if file_path.exists():
            self.log("Step3 results loaded.")
        else:
            self.log("Step3 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                symbols = load_symbols_from_str(False, row["symbols"])
                if row["target_variable"] == "":
                    target_variable = None
                else:
                    if row["target_variable"] in symbols:
                        target_variable = symbols[row["target_variable"]]
                    else:
                        row["target_variable"] = None

                triplets = str_to_triplets(False, row["triplets"], symbols)

                r = Step3RegexResult(row["target_type"], triplets, symbols, target_variable, row["errors"],
                                     bool(int(row["target_var_not_in_triplets"])))
                r.num_valid_variables = row["num_variables"]
                r.num_valid_constants = row["num_constants"]
                r.target_type_is_invalid = bool(int(row["tt_invalid"]))
                if r.target_type_is_invalid:
                    r.target_type = None
                else:
                    r.target_type = row["target_type"]
                r.target_type_ground_truth = row["tt_ground_truth"]
                r.target_type_is_incorrect = bool(int(row["tt_incorrect"]))

                self.step3_results[int(row["q_id"])] = r

    def save_step4(self, question_id: int, query: str, r: Step4SymbolCandidatesResult):
        self.step4_results[question_id] = r
        r.save_to_file(question_id, query, self.output_path)

    def load_step4(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step4_symbol_candidates.csv"
        else:
            file_path = Path(overwrite_path) / "step4_symbol_candidates.csv"

        if file_path.exists():
            self.log("Step4 results loaded.")
        else:
            self.log("Step4 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                r = Step4SymbolCandidatesResult(bool(int(row["skipped"])))
                r.valid_symbols = None  # will be lazy loaded when used because of its larger size
                self.step4_results[int(row["q_id"])] = r

    def save_step5(self, question_id: int, r: Step5GroundTripletsResult, ground_truths: list[int]):
        self.step5_results[question_id] = r
        r.save_to_file(question_id, self.output_path, ground_truths)

    def load_step5(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step5_ground_triplets.csv"
        else:
            file_path = Path(overwrite_path) / "step5_ground_triplets.csv"

        if file_path.exists():
            self.log("Step5 results loaded.")
        else:
            self.log("Step5 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["skipped"]))
                if skipped:
                    answers_flattened = set()
                    answers = [set()]
                else:
                    answers_flattened = ast.literal_eval(row["answers_flattened"])
                    answers = ast.literal_eval(row["answers"])
                r = Step5GroundTripletsResult(answers, answers_flattened, int(row["variables_without_candidates"]),
                                              int(row["variable_candidates"]), int(row["l_first_hit"]),
                                              int(row["l_last_hit"]), skipped)

                r.num_true_pos_in_prefilter = int(row["true_pos"])
                r.num_false_pos_in_prefilter = int(row["false_pos"])
                r.precision = float(row["precision"])
                r.recall = float(row["recall"])
                r.num_target_candidates = int(row["target_candidates"])
                self.step5_results[int(row["q_id"])] = r

    def save_step6(self, question_id: int, r: Step6VSSResult):
        self.step6_results[question_id] = r

        answers_vss = r.vss_top_hits
        vss_scores = r.vss_scores

        metrics = calculate_metrics(answers_vss, r.ground_truths)

        res_file_path = self.output_path / "step6_vss.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,hit_1,hit_5,hit_20,hit_40,hit_50,reci_rank,reci_rank_20,recall_20,fallback,"
                    "answers_vss,vss_scores\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, metrics["hit_1"], metrics["hit_5"], metrics["hit_20"], metrics["hit_40"],
                             metrics["hit_50"], metrics["reciprocal_rank"],
                             metrics["reciprocal_rank_20"], metrics["recall_20"], int(r.fallback_solution),
                             answers_vss, vss_scores])

    def load_step6(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step6_vss.csv"
        else:
            file_path = Path(overwrite_path) / "step6_vss.csv"

        if file_path.exists():
            self.log("Step6 results loaded.")
        else:
            self.log("Step6 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["fallback"]))
                vss_top_hits = ast.literal_eval(row["answers_vss"])
                vss_scores = ast.literal_eval(row["vss_scores"])
                r = Step6VSSResult(vss_top_hits, vss_scores, skipped)

                self.step6_results[int(row["q_id"])] = r

    def save_step6_plus_7(self, question_id: int, r: Step6plus7VSSResult):
        self.step6_plus_7_results[question_id] = r

        answers_vss = r.vss_top_hits
        vss_scores = r.vss_scores

        metrics = calculate_metrics(answers_vss, r.ground_truths)

        res_file_path = self.output_path / "step6_and_7_vss.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,hit_1,hit_5,hit_20,hit_40,hit_50,reci_rank,reci_rank_20,recall_20,"
                    "answers_vss,vss_scores,answer_sources\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, metrics["hit_1"], metrics["hit_5"], metrics["hit_20"], metrics["hit_40"],
                             metrics["hit_50"], metrics["reciprocal_rank"],
                             metrics["reciprocal_rank_20"], metrics["recall_20"],
                             answers_vss, vss_scores, r.answer_sources])

    def load_step6_and_7(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step6_and_7_vss.csv"
        else:
            file_path = Path(overwrite_path) / "step6_and_7_vss.csv"

        if file_path.exists():
            self.log("Step6_and_7 results loaded.")
        else:
            self.log("Step6_and_7 results not existing yet.")
            return

        with (open(file_path, 'r', encoding='utf-8', newline='') as result_file):
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                vss_top_hits = ast.literal_eval(row["answers_vss"])
                vss_scores = ast.literal_eval(row["vss_scores"])
                answer_sources = ast.literal_eval(row["answer_sources"])
                r = Step6plus7VSSResult(None, None, 0, 0)
                r.vss_top_hits = vss_top_hits
                r.vss_scores = vss_scores
                r.answer_sources = answer_sources

                self.step6_plus_7_results[int(row["q_id"])] = r

    def save_step7(self, question_id: int, r: Step7VSSResult):
        self.step7_results[question_id] = r
        answers_vss = r.vss_top_hits
        vss_scores = r.vss_scores

        metrics = calculate_metrics(answers_vss, r.ground_truths)

        res_file_path = self.output_path / "step7_vss.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,hit_1_vss,hit_5_vss,hit_20_vss,hit_40_vss,hit_50_vss,reci_rank_vss,reci_rank_20_vss,recall_20_vss,fallback,"
                    "answers_vss,vss_scores\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([question_id, metrics["hit_1"], metrics["hit_5"], metrics["hit_20"], metrics["hit_40"],
                             metrics["hit_50"], metrics["reciprocal_rank"],
                             metrics["reciprocal_rank_20"], metrics["recall_20"],
                             int(r.fallback_solution),
                             answers_vss, vss_scores])

    def load_step7(self, overwrite_path=None):
        if overwrite_path is None:
            file_path = self.output_path / "step7_vss.csv"
        else:
            file_path = Path(overwrite_path) / "step7_vss.csv"

        if file_path.exists():
            self.log("Step7 results loaded.")
        else:
            self.log("Step7 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["fallback"]))
                vss_top_hits = ast.literal_eval(row["answers_vss"])
                vss_scores = ast.literal_eval(row["vss_scores"])
                r = Step7VSSResult(vss_top_hits, vss_scores, skipped)

                self.step7_results[int(row["q_id"])] = r

    def save_step8(self, question_id: int, r: Step8FinalRerankerResult, question: str):
        self.step8_results[question_id] = r
        answers = r.answer_ids

        metrics = calculate_metrics(answers, r.ground_truths)

        res_file_path = self.output_path / "step8_final_answers.csv"
        if not res_file_path.exists():
            with open(res_file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write(
                    "q_id,question,final_answer,ground_truth,hit_1,hit_5,hit_20,hit_40,hit_50,reci_rank,reci_rank_20,recall_20,"
                    "fallback,answers\n")

        with open(res_file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [question_id, question, r.final_answer_str, r.ground_truth_str, metrics["hit_1"], metrics["hit_5"],
                 metrics["hit_20"], metrics["hit_40"], metrics["hit_50"], metrics["reciprocal_rank"],
                 metrics["reciprocal_rank_20"], metrics["recall_20"], int(r.fallback_solution),
                 answers])

    def load_step8(self):
        file_path = self.output_path / "step8_final_answers.csv"

        if file_path.exists():
            self.log("Step8 results loaded.")
        else:
            self.log("Step8 results not existing yet.")
            return

        with open(file_path, 'r', encoding='utf-8', newline='') as result_file:
            reader = csv.DictReader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for row in reader:
                skipped = bool(int(row["fallback"]))
                answer_ids = ast.literal_eval(row["answers"])
                r = Step8FinalRerankerResult(answer_ids, row["final_answer"], skipped)

                self.step8_results[int(row["q_id"])] = r

    def log_llm_costs(self, qid: int, step: int, num_input_tokens: int, num_output_tokens: int, estimated_costs: float):
        file_path = self.output_path / "llm_costs.csv"
        if not file_path.exists():
            with open(file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,step,input_tokens,output_tokens,expected_costs\n")

        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([qid, step, num_input_tokens, num_output_tokens, round(estimated_costs, 6)])

    def log_compute_time(self, qid: int, step: int, compute_time: float):
        file_path = self.output_path / "run_times.csv"
        if not file_path.exists():
            with open(file_path, 'x', encoding='utf-8', newline='') as result_file:
                result_file.write("q_id,step,seconds\n")

        with open(file_path, 'a', encoding='utf-8', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([qid, step, round(compute_time,6)])