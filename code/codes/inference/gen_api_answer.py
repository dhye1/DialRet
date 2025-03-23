import argparse
import time
import os
import json
from tqdm import tqdm
from openai import OpenAI


def run_eval(model_id, question_file, answer_file):
	ques_jsons = []
	with open(question_file, "r") as ques_file:
		for line in ques_file:
			ques_jsons.append(line)
		ques_jsons = ques_jsons[:100]
	
	ans_jsons = []
	for i, line in enumerate(tqdm(ques_jsons)):
		ques_json = json.loads(line)
		client = OpenAI()
		response = client.chat.completions.create(
			model=model_id,
			messages=[{"role": "user", "content": ques_json["input"]
			}]
		)
		outputs = response.choices[0].message.content
		ans_jsons.append(
				{
					"question_id": i,
					"text": outputs,
					"model_id": model_id,
					"metadata": {},
				}
		)
	os.makedirs('/'.join(answer_file.split('/')[:-1]), exist_ok=True)
	with open(os.path.expanduser(answer_file), "w") as ans_file:
		for line in ans_jsons:
				ans_file.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model-id", type=str, required=True)
	parser.add_argument("--question-file", type=str, required=True)
	parser.add_argument("--answer-file", type=str, default="answer.jsonl")
	args = parser.parse_args()

	run_eval(
		args.model_id,
		args.question_file,
		args.answer_file,
	)