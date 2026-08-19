import os
import json
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer
import evaluate
import numpy as np
import pickle
import gc
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

def apply_chat_template(model_name, prompts):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Answer the following question. Output your answer in <reasoning> </reasoning> and <answer> </answer> tags.\n<question> {p} </question>"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

LAJ_MATH_PROMPT = """You are an expert mathematical reasoning evaluator.

You will be given a reference answer and a predicted answer. Determine whether they are mathematically equivalent.

Respond using exactly this format:
<reasoning>your step-by-step reasoning</reasoning>
<score>0 or 1</score>"""

LAJ_ALPACA_PROMPT = """You are an expert LLM-as-a-Judge evaluator.

You will be given a reference answer and a predicted answer. Determine whether they match in meaning and correctness.

Respond using exactly this format:
<reasoning>your step-by-step reasoning</reasoning>
<score>0 or 1</score>"""



def parse_laj_score(text):
    m = re.search(r'<score>\s*([01])\s*</score>', text)
    if m:
        return float(m.group(1))
    try:
        return float(json.loads(text)["score"])
    except Exception:
        pass
    m = re.search(r'\b([01])\b', text[::-1])
    if m:
        return float(m.group(1))
    with open('laj_parse_errors.txt', 'a+') as f:
        f.write(f"{text}\n\n")
    0/0

def evaluate_laj(llm, answers, predictions, laj_prompt):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
    raw_prompts = [f"{laj_prompt}\nReference Answer: {a}\nPredicted Answer: {p}" for a, p in zip(answers, predictions)]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]
    laj_sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    outputs = llm.generate(prompts, sampling_params=laj_sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]
    scores = torch.tensor([parse_laj_score(o) for o in outputs])
    return scores.tolist()

def evaluate_rouge(rouge_metric, answers, outputs):
    rouge_scores = []
    for a, o in zip(answers, outputs):
        rouge = rouge_metric.compute(predictions=[o], references=[a])["rougeL"]
        rouge_scores.append(1.0 if rouge > 0.5 else 0.0)
    return rouge_scores

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default="qwen7b")
    args = parser.parse_args()

    with open(f'result_files/{args.exp_name}.pkl', 'rb') as f:
        evaluation_settings = pickle.load(f)

    ### EVALUATING ALL LAJ BENCHMARKS ###
    llm = LLM(
        "Qwen/Qwen2.5-32B-Instruct",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=4096,
        dtype="bfloat16",
        max_num_seqs=32,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
    for eval in evaluation_settings.keys():
        print("Evaluating LAJ for", eval)
        if evaluation_settings[eval]["metric"] != "laj":
            continue
        answers = evaluation_settings[eval]["answers"]
        all_outputs = evaluation_settings[eval]["outputs"]

        all_scores = []
        for outputs in all_outputs:
            scores = evaluate_laj(llm, answers, outputs, evaluation_settings[eval]["laj_prompt"])
            all_scores.append(scores)
        evaluation_settings[eval]["scores"] = all_scores

    del llm, sampling_params
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()
    ### EVALUATING ALL LAJ BENCHMARKS ###




    ### EVALUATING ALL ROUGE BENCHMARKS ###
    rouge_metric = evaluate.load('rouge')
    for eval in evaluation_settings.keys():
        print("Evaluating ROUGE for", eval)
        if evaluation_settings[eval]["metric"] != "rouge":
            continue
        answers = evaluation_settings[eval]["answers"]
        all_outputs = evaluation_settings[eval]["outputs"]

        all_scores = []
        for outputs in all_outputs:
            scores = evaluate_rouge(rouge_metric, answers, outputs)
            all_scores.append(scores)
        evaluation_settings[eval]["scores"] = all_scores
    del rouge_metric
    ### EVALUATING ALL ROUGE BENCHMARKS ###


    with open(f"result_files/{args.exp_name}.pkl", 'wb') as f:
        pickle.dump(evaluation_settings, f)

    ### OUTPUTTING RESULTS ###
    for eval in evaluation_settings.keys():
        scores = evaluation_settings[eval]["scores"]
        with open('results.txt', 'a+') as f:
            mean_scores = np.array(scores).mean(axis=1)
            mean = np.mean(mean_scores)
            std = np.std(mean_scores)
            f.write(f"{args.exp_name}\t{eval}\t{mean}\t{std}\t{mean_scores}\n")
    ### OUTPUTTING RESULTS ###