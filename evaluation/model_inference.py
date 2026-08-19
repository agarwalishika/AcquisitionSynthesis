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

def generate_outputs(llm, sampling_params, args, prompts, num_rounds):
    all_outputs = []
    for i in range(num_rounds):
        prompts = apply_chat_template(args.model_name, prompts)

        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]

        outputs = [output.split("<answer>")[-1].replace("</answer>", "").strip() for output in outputs]
        all_outputs.append(outputs)
    return all_outputs

def evaluate_outputs(predictions, references):
    embedding_model = LLM('Qwen/Qwen3-Embedding-0.6B', task="embed")

    pred_embs = embedding_model.embed(predictions)
    pred_embs = torch.tensor([o.outputs.embedding for o in pred_embs])

    ref_embs = embedding_model.embed(references)
    ref_embs = torch.tensor([o.outputs.embedding for o in ref_embs])

    scores = (pred_embs @ ref_embs.T).diag()

    return scores

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
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir_path", default="/home/ishikaa2/AcquisitionSynthesis/data")
    parser.add_argument("--num_rounds", default=1, type=int)
    args = parser.parse_args()

    DATA_DIR_PATH = args.data_dir_path

    evaluation_settings = {
        "numina": {"path": os.path.join(DATA_DIR_PATH, "numina/test.parquet"), "metric": "laj", "laj_prompt": LAJ_MATH_PROMPT},
        # "medmcqa": {"path": os.path.join(DATA_DIR_PATH, "medmcqa/test.parquet"), "metric": "rouge"},
        # "aime": {"path": os.path.join(DATA_DIR_PATH, "aime/test.parquet"), "metric": "laj", "laj_prompt": LAJ_MATH_PROMPT},
        # "pubmedqa": {"path": os.path.join(DATA_DIR_PATH, "pubmedqa/test.parquet"), "metric": "rouge"},
        # "alpaca": {"path": os.path.join(DATA_DIR_PATH, "alpaca/test.parquet"), "metric": "laj", "laj_prompt": LAJ_ALPACA_PROMPT},
    }

    ### GENERATING OUTPUTS ###
    llm = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    for eval in evaluation_settings.keys():
        print("Generating outputs for", eval, args.model_name)
        grounding_seed = pd.read_parquet(evaluation_settings[eval]["path"], engine='pyarrow')
        prompts = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))
        answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))

        outputs = generate_outputs(llm, sampling_params, args, prompts, args.num_rounds)
        evaluation_settings[eval]["outputs"] = outputs
        evaluation_settings[eval]["prompts"] = prompts
        evaluation_settings[eval]["answers"] = answers

    destroy_model_parallel()
    destroy_distributed_environment()
    # del llm.llm_engine.model_executor
    del llm, sampling_params
    gc.collect()
    torch.cuda.empty_cache()

    ### GENERATING OUTPUTS ###



    ### EVALUATING ALL LAJ BENCHMARKS ###
    llm = LLM("Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.9, trust_remote_code=True, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
    for eval in evaluation_settings.keys():
        print("Evaluating LAJ for", eval, args.model_name)
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
    ### EVALUATING ALL LAJ BENCHMARKS ###




    ### EVALUATING ALL ROUGE BENCHMARKS ###
    rouge_metric = evaluate.load('rouge')
    for eval in evaluation_settings.keys():
        print("Evaluating ROUGE for", eval, args.model_name)
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



    ### OUTPUTTING RESULTS ###
    for eval in evaluation_settings.keys():
        answers = evaluation_settings[eval]["answers"]
        prompts = evaluation_settings[eval]["prompts"]
        outputs = evaluation_settings[eval]["outputs"]
        scores = evaluation_settings[eval]["scores"]

        exp_name = f"{args.model_name.split('/')[-1]}_{eval}"
        prompts = [p.replace("\n", " ") for p in prompts]
        answers = [a.replace("\n", " ") for a in answers]
        # outputs = [o.replace("\n", " ") for o in outputs for outputs in all_outputs]

        results = {
            "question": prompts,
            "predictions": outputs,
            "references": answers,
            "scores": scores
        }
        
        with open(f"result_files/{exp_name}.pkl", 'wb') as f:
            pickle.dump(results, f)

        with open('results.txt', 'a+') as f:
            mean_scores = np.array(scores).mean(axis=1)
            mean = np.mean(mean_scores)
            std = np.std(mean_scores)
            f.write(f"{exp_name}\t{mean}\t{std}\t{mean_scores}\n")
    ### OUTPUTTING RESULTS ###
