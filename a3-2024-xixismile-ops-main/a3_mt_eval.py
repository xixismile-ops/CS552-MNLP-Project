import os
import evaluate
import json
from tqdm import tqdm
import numpy as np # NOTE: you don't have to use it but you are allowed to
import pandas as pd
from itertools import combinations
from tqdm import tqdm

def load_json(filename):
    """Helper to load JSON files."""
    with open(filename, 'r', encoding='utf-8') as read_file:
        data = json.load(read_file)
    return data

def save_json(mydictlist, filename):
    """Helper to save JSON files."""
    f = open(filename, 'w', encoding='utf-8')
    json.dump(mydictlist, f, ensure_ascii=False, indent=4) 
    f.close()

def create_entryid2score(entry_ids, scores):
    """Zips entry IDs and scores and creates a dictionary out of this mapping.

    Args:
        entry_ids (str list): list of data entry IDs
        scores (float list): list of scores

    Returns:
        dict: given a list of aligned entry IDs and scores creates a dictionary 
                that maps from an entry ID to the corresponding score

    """
    score_dict = {}
    for entry_id, res in zip(entry_ids, scores):
        score_dict[str(entry_id)] = res
    return score_dict

def calculate_metrics():
    ############################################################################
    # 1) Load data
    ############################################################################
    wmt_da_df = pd.read_csv(os.path.join("data", "wmt-da-human-evaluation_filtered.csv"))
    
    ############################################################################
    # 2) Load HF metrics
    ############################################################################
    # TODO: load the BLEU, BERTScore and COMET metrics from the evaluate package
    ############################################################################
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    comet = evaluate.load("comet")
    
    # NOTE: of the form {metric_name: {entry_id: score, ...}, ...}
    metric_dict = {}
        
    ############################################################################
    # 3.1) Calculate BLEU    
    ############################################################################
    # TODO: calculate the following bleu scores for each hypothesis
    #       - BLEU
    #       - BLEU-1
    #       - BLEU-4
    # Make sure to populate the metric_dict dictionary for each of these scores:
    #   For example, for BLEU-1 the metric_dict entry should look like:
    #   metric_dict["bleu-1"] = { "23423": 0.5 } where "23423" is an entry_id 
    #   and 0.5 is the BLEU-1 score it got. 
    # 
    # Feel free to use the `create_entryid2score` helper function.
    # https://huggingface.co/spaces/evaluate-metric/bleu
    ############################################################################
    print("-" * 50)
    print("Calculating BLEU...")
    metric_dict["bleu"] = {}
    metric_dict["bleu-1"] = {}
    metric_dict["bleu-4"] = {}

    for _, row in wmt_da_df.iterrows():
        mt_output = row['mt']
        ref_output = [row['ref']]

        bleu_result = bleu.compute(predictions=[mt_output], references=[ref_output])
        metric_dict["bleu"][row["entry_id"]] = bleu_result["bleu"]
        metric_dict["bleu-1"][row["entry_id"]] = bleu_result["precisions"][0]
        metric_dict["bleu-4"][row["entry_id"]] = bleu_result["precisions"][-1]

    print("Done.")
        
    ############################################################################
    # 3.2) Calculate BERTScore
    ############################################################################
    # TODO: calculate the following BERTScore-s for each hypothesis
    #       - Precision
    #       - Recall
    #       - F-1
    # Make sure to populate the metric_dict dictionary for each of these scores.
    # Feel free to use the `create_entryid2score` helper function.
    #
    # For BERTScore, you will require to pass a `lang` parameter. Please read 
    # the documentation to figure out what that might mean. 
    # (Hint: For `lang`, you may want to use the `groupby` function of pandas dataframes!)
    # https://huggingface.co/spaces/evaluate-metric/bertscore
    ############################################################################
    print("-" * 50)
    print("Calculating BERTScore...")
    metric_dict["bertscore-precision"] = {}
    metric_dict["bertscore-recall"] = {}
    metric_dict["bertscore-f1"] = {}

    for (lang_pair, group) in wmt_da_df.groupby('lp'):
        lang = lang_pair.split('-')[0] 

        for _, row in group.iterrows():
            mt_output = row['mt']
            ref_output = [row['ref']]

            bertscore_results = bertscore.compute(predictions=[mt_output], references=[ref_output], lang=lang)

            entry_id = row["entry_id"]
            metric_dict["bertscore-precision"][entry_id] = bertscore_results["precision"][0]
            metric_dict["bertscore-recall"][entry_id] = bertscore_results["recall"][0]
            metric_dict["bertscore-f1"][entry_id] = bertscore_results["f1"][0]
    
    print("Done.")
    
    ############################################################################
    # 3.3) Calculate COMET
    ############################################################################
    # TODO: calculate the COMET score for each hypothesis
    # Make sure to populate the metric_dict dictionary for COMET.
    # Feel free to use the `create_entryid2score` helper function.
    # https://huggingface.co/spaces/evaluate-metric/comet
    ############################################################################
    print("-" * 50)
    print("Calculating COMET...")
    metric_dict["comet"] = {}

    predictions = []
    references = []
    sources = []
    entry_ids = []

    for _, row in wmt_da_df.iterrows():
        predictions.append(row['mt'])
        references.append(row['ref'])
        sources.append(row['src'])
        entry_ids.append(row['entry_id'])
    
    comet_results = comet.compute(predictions=predictions,
                              references=references,
                              sources=sources,
                              progress_bar=True)
                              
    for i, entry_id in enumerate(entry_ids):
        metric_dict['comet'][entry_id] = comet_results['scores'][i]

    print("Done.")
    
    ############################################################################
    # 4) Save the output in a JSON file
    ############################################################################
    save_json(metric_dict, "part3_metrics.json")
    return metric_dict
    

def evaluate_metrics():
    ############################################################################
    # 1) Load data
    ############################################################################
    wmt_da_df = pd.read_csv(os.path.join("data", "wmt-da-human-evaluation_filtered.csv"))
    print(wmt_da_df.head())
    print(len(wmt_da_df))
    
    ############################################################################
    # 2) Create ranked data for Kendall's Tau
    ############################################################################
    # TODO: For each (source, lp) group, rank the entry_id s by the "score".
    #       And then create rank_pairs_list: a list of ranking pairs which are
    #       (worse hypothesis id, better_hypothesis id)
    #       Hint: use combinations from itertools!
    ############################################################################  
    rank_pairs_list = []

    ranked_data = wmt_da_df.groupby(['src', 'lp']).apply(
        lambda x: x.sort_values('src').reset_index(drop=True)
        ).reset_index(drop=True)

    for (src, lp), group in ranked_data.groupby(['src', 'lp']):
        for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
            worse_hypothesis_id = row1['entry_id'] if row1['score'] < row2['score'] else row2['entry_id']
            better_hypothesis_id = row2['entry_id'] if row1['score'] < row2['score'] else row1['entry_id']
            rank_pairs_list.append((worse_hypothesis_id, better_hypothesis_id))

    # NOTE: The following should be ~3351
    print("Size of rank combinations: ", len(rank_pairs_list))
    
    ############################################################################
    # 2) Create a class to calculate Kendalls Tau for each metric
    ############################################################################
    # TODO: Complete the class such that each call to the class can update the 
    #       count of concordant and discordant values
    ############################################################################
    class KendallsTau:
        """
        A class to accumulate concordant and discordant instances and to
        compute Kendall's Tau correlation coefficient. 
        Helps when iteratively doing the computation.
        Feel free to implement it otherwise if you don't want to do it iteratively.
        """
        def __init__(self):
            self.concordant = 0.0
            self.discordant = 0.0
            self.total = 0.0

        def update(self, worse_hyp_score, better_hyp_score):
            """Updates the concordant and discordant values.

            Args:
                worse_hyp_score (float): the score for the worse hypothesis 
                        according to human ranking
                better_hyp_score (float): the score for the better hypothesis 
                        according to human ranking
            """
            if worse_hyp_score < better_hyp_score:
                self.concordant += 1
            elif worse_hyp_score > better_hyp_score:
                self.discordant += 1
            self.total += 1

        def compute(self):
            """
            Calculates the Kendall's Tau correlation coefficient. 
            Call when all ranked pairs have been evaluated.
            """
            tau = (self.concordant - self.discordant) / self.total if self.total != 0 else 0
            return tau
    
    ############################################################################
    # 3) Calculate Kendall's Tau correlation for each metric
    ############################################################################
    # TODO: Populate the metrics2kendalls dictionary s.t. we get 
    #       metric2kendalls = {metric_name: correlation, ...} for all metrics
    ############################################################################
    metric_dict = load_json("part3_metrics.json")
    metric2kendalls = {}

    for metric_name in metric_dict:
        metric2kendalls[metric_name] = KendallsTau()
    
    for worse_hypothesis_id, better_hypothesis_id in tqdm(rank_pairs_list):
        for metric_name in metric_dict.keys():
            worse_hypothesis_id = str(worse_hypothesis_id)
            better_hypothesis_id = str(better_hypothesis_id)

            worse_hypothesis_score = metric_dict[metric_name][worse_hypothesis_id]
            better_hypothesis_score = metric_dict[metric_name][better_hypothesis_id]
            metric2kendalls[metric_name].update(worse_hypothesis_score, better_hypothesis_score)

    for metric_name in metric_dict.keys():
        metric2kendalls[metric_name] = metric2kendalls[metric_name].compute()
    
    ############################################################################
    # 4) Save the output in a JSON file
    ############################################################################
    save_json(metric2kendalls, "part3_corr.json")
        

if __name__ == '__main__':
    already_predicted_scores = False
    if not already_predicted_scores:
        calculate_metrics()
    evaluate_metrics()
