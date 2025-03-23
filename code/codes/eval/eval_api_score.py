import argparse
import os
from tqdm import tqdm
from openai import OpenAI

import os
import re
import json
import argparse
import numpy as np
import pandas as pd

from evaluate import load
from perplexity import Perplexity
import ast

class Metrics:
    def __init__(self):
        pass


    def _get_api_score(self, predictions, references, contexts):
        client = OpenAI()

        output_json = []
        output_ratings = []
        
        for i, (pred, ref, context) in enumerate(zip(tqdm(predictions), references, contexts)):
            
            prompt="""Please act as an impartial judge and evaluate the quality of the last response of given multi-session conversation in terms of coherence and consistency.
You are provided with <reference response> and <assistant response> which will be included in the last dialogue session ###.
Begin your evaluation by comparing the <assistant's response> and the <reference response> by referring to previous dialogue sessions.
As objectively as possible, <assistant's response>  should be evaluated with a focus on iengagingness, humanness, and memorability.
After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format:
"[[rating]]", for example: "Rating: [[5]]".\n"""
            
            if i==20:
                break
                
            eval_input = prompt + context + "\n<assistant's response>:" + pred + "\n<reference response>:" + ref
            print(eval_input)
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    # model="gpt-4-1106-preview",
                    # response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": str(eval_input)}],
                )
                eval_output = response.choices[0].message.content
                print(eval_output)
                
                match = re.search(r'\[\[(\d+)\]\]', eval_output)
                eval_json ={}
                try:
                    rating = int(match.group(1))
                    eval_json['id'] = i
                    eval_json['rating']=rating
                    eval_json['explanation']=eval_output
                    eval_json['prediction'] = pred.rstrip('\n')
                    eval_json['reference'] = ref.rstrip('\n')
                # eval_json['context'] = context
                except:
                    rating = None

                output_json.append(eval_json)
                output_ratings.append(eval_json['rating'])
            except:
                print(f'Error: {i}th evaluation')
            

        gpt4score = np.mean(output_ratings)
        output_json_df = pd.DataFrame(output_json)
        return gpt4score, output_json_df

    def _save_eval_reason_json(self, output_json_df, output_path, model_id, tuned, cross_domain,):
        os.makedirs(f"{args.output_path}_reason", exist_ok=True)
        output_json_df.to_csv(f'{output_path}_reason/{model_id}_{tuned}+{cross_domain}.csv', index=False)

    def calculate_metrics(self, predictions, references, contexts, output_path, model_id, tuned, cross_domain):
        gpt4score, output_json_df = self._get_api_score(predictions, references, contexts)
        self._save_eval_reason_json(output_json_df, output_path, model_id, tuned, cross_domain)
        return {'GPT4Score': round(gpt4score,4)}

def str_to_list(s):
    v = s.split(',')
    return v

import torch
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--model-tuned", type=str_to_list, required=True)
    parser.add_argument("--cross-domain", type=str, required=True)
    parser.add_argument("--reference-path", type=str, required=True)
    parser.add_argument("--inference-path", type=str, required=True)
    parser.add_argument("--context-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--cpu-thread-num", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_thread_num)

    metrics = Metrics()
    metric_result_list = []


    for tuned in args.model_tuned:
        # added context-path
        print(f"{args.context_path}/{tuned}/test.jsonl")

        if args.cross_domain=="indomain":
            df_context = pd.read_json(f"{args.context_path}/{tuned}/test.jsonl", lines=True, orient='records')
        else:
            df_context = pd.read_json(f"{args.context_path}/{args.cross_domain}/test.jsonl", lines=True, orient='records')
        contexts = df_context['input'].to_list()
        
        # same as eval_bertscore.py
        print(f"{args.inference_path}_{args.cross_domain}/inf_{args.model_id}_{tuned}.jsonl")
        print(f"{args.reference_path}/{tuned}.jsonl")
        df_pred = pd.read_json(f"{args.inference_path}_{args.cross_domain}/inf_{args.model_id}_{tuned}.jsonl", lines=True, orient='records')
        if args.cross_domain=="indomain":
            df_ref = pd.read_json(f"{args.reference_path}/{tuned}.jsonl", lines=True, orient='records')
        else:
            df_ref = pd.read_json(f"{args.reference_path}/{args.cross_domain}.jsonl", lines=True, orient='records')

        predictions = df_pred['text'].tolist()
        references = df_ref['output'].to_list()

        metric_score = metrics.calculate_metrics(predictions, references, contexts, args.output_path, args.model_id, tuned, args.cross_domain)
        metric_score['Model']=f'{args.model_id}_{tuned}'
        metric_result_list.append(metric_score)

    result = pd.DataFrame(metric_result_list)
    result = result[['Model', 'GPT4Score']]
    os.makedirs(args.output_path, exist_ok=True)
    result.to_csv(f'{args.output_path}/{args.model_id}+{args.cross_domain}.csv', index=False)
    print(result)
    
    
#                 prompt="""You are a helpful assistant designed to output JSON. 
# Please act as an impartial judge and evaluate the quality of the response provided multi-session dialogue.
# Your evaluation should consider coherence and consistency. You will be given a <reference response> and the <assistant's response>.
# You evaluation should focus on the <assistant's response> to the last dialogue session.
# Begin your evaluation by comparing the <assistant's response> with the <reference response>.
# Evaluate focus on engagingness, humanness, memorability response as objective as possible.
# After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format:
# "{'rating': ,'explanation':}", for example: "{'rating': 5,'explanation': 'because this...'}".\n"""