import os
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from evaluate import load
from perplexity import Perplexity
import ast

class Metrics:
    def __init__(self):
        self.bertscore = load("bertscore")
        self.rouge = load("rouge")
        self.perp = Perplexity()

    def _get_ppl(self, predictions, references, model_id):
        ## For OOM
        # result_list = []
        # for pred, label in zip(predictions, references):
        #     result = self.perp.compute(data=label, model_id='pinkmanlove/llama-7b-hf')
        #     result_list.append(result['perplexities'])
        # return np.array(result_list).mean()
        # At once
        result = self.perp.compute(data=predictions, model_id='pinkmanlove/llama-7b-hf', batch_size=1)#, device="cpu")
        return result['mean_perplexity']

    def _get_rouge(self, predictions, references):
        result = self.rouge.compute(predictions=predictions,references=references)
        return result

    def _get_bertscore(self, predictions, references):
        ## For OOM
        # result_list = []
        # for pred, label in zip(predictions, references):
        #     result = self.bertscore.compute(predictions=[pred], references=[label], model_type="microsoft/deberta-xlarge-mnli")
        #     result_list.append(result['f1'])
        # return np.array(result_list).mean()
        ## At once
        result = self.bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli", batch_size=100, device='cpu')
        return np.array(result['f1']).mean()

    def calculate_metrics(self, predictions, references, model_id):
        # ppl_s = self._get_ppl(predictions, references)
        bert_s = self._get_bertscore(predictions, references)
        rouge_s = self._get_rouge(predictions, references)
        # print('ppl:', ppl_s)
        print('bertscore:', bert_s)
        print('rouge:', rouge_s)
        # return {'PPL':ppl_s,  'ROUGE':rouge_s, 'BertScore':bert_s}
        return {'ROUGE-1': round(rouge_s['rouge1'],4), 'ROUGE-2': round(rouge_s['rouge2'],4), 'ROUGE-L': round(rouge_s['rougeL'],4), 'BERT-S':round(bert_s,4)}

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
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--cpu-thread-num", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_thread_num)
    
    metrics = Metrics()
    metric_result_list = []
 
    for tuned in args.model_tuned:
        print(f"{args.inference_path}_{args.cross_domain}/inf_{args.model_id}_{tuned}.jsonl")
        df_pred = pd.read_json(f"{args.inference_path}_{args.cross_domain}/inf_{args.model_id}_{tuned}.jsonl", lines=True, orient='records')
        if args.cross_domain=="indomain":
            print(f"{args.reference_path}/{tuned}.jsonl")
            df_ref = pd.read_json(f"{args.reference_path}/{tuned}/test.jsonl", lines=True, orient='records')
        else:
            print(f"{args.reference_path}/{args.cross_domain}/test.jsonl")
            df_ref = pd.read_json(f"{args.reference_path}/{args.cross_domain}/test.jsonl", lines=True, orient='records')

        predictions = df_pred['text'].tolist()
        references = df_ref['output'].to_list()

        metric_score = metrics.calculate_metrics(predictions, references, args.model_id)
        metric_score['Model']=f'{args.model_id}_{tuned}'
        metric_result_list.append(metric_score)

    result = pd.DataFrame(metric_result_list)
    result = result[['Model', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT-S']]

    output_file = f'{args.output_path}/{args.model_id}+{args.cross_domain}_BS.csv'
    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_file, index=False)
    print(result)