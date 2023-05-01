'''Generate a table of performance summary for all model configurations on all datasets in output_dir/.'''
import os
if os.getcwd().endswith("evaluate"):
	os.chdir("../..")  # change the working directory to the root directory
import sys
sys.path.append("source")
from dataset.utils import load_data
from evaluate.evaluate_answer_acc import evaluate_acc
import jsonlines
import pandas as pd

if __name__ == "__main__":
	performance_summary = {}

	output_dir = "output_dir"
	split = "test"

	for dataset_name in os.listdir(output_dir):
		dataset_frn = f"data/{dataset_name}/{split}.jsonl"

		if not os.path.exists(dataset_frn):
			continue

		dataset = load_data(dataset_frn)
		performance_summary[dataset_name] = {}

		preds_dir = f"{output_dir}/{dataset_name}/{split}"
		for model_name in os.listdir(preds_dir):
			if model_name in ["performance_summary.csv", ".DS_Store"]:
				continue
			pred_file = f"{preds_dir}/{model_name}/predictions.jsonl"
			if not os.path.exists(pred_file):
				continue
			with open(pred_file, "r") as fr:
				reader = jsonlines.Reader(fr)
				predictions = list(reader)
			try:
				acc = evaluate_acc(dataset=dataset,
				                   predictions=predictions,
				                   dataset_name=dataset_name,
				                   )
			except:
				print(f"Failed to evaluate {dataset_name} {model_name}")
				acc = "N/A"
				continue
			performance_summary[dataset_name][model_name] = acc

	# save the performance summary as a pandas dataframe
	df = pd.DataFrame(performance_summary)

	# sort the rows
	row_names = [
		"code002_NL+SL",
		"code002_NL+SL_n:40",
		"code002_NL+SL_exemplarset1",
		"code002_NL+SL_exemplarset2",
		"code002_NL+SL_exemplarset3",
		"code002_NL+SL_exemplarset4",
		"code002_NL+SL_exemplarset5",
		"code002_prompt_variation1",
		"code002_prompt_variation2",
		"code002_prompt_variation3",
		"code002_norationale",
		"code002_noNLbutnudge",
		"code002_nosolver",
		"code002_noNL",
		"text001_NL+SL",
		"text002_NL+SL",
		"text003_NL+SL",
		"code001_NL+SL",
		"gpt-3.5-turbo_NL+SL",
		"gpt4_NL+SL"
	]
	df = df.reindex(row_names)

	column_names = [
		"GSM8K",
		"SVAMP",
		"MultiArith",
		"ASDiv",
		"AQUA",
		"saycan",
		"StrategyQA",
		"date",
		"sports",
		"CLUTRR"
	]
	# sort the columns
	df = df[column_names]

	df.to_csv(f"{output_dir}/performance_summary.csv")