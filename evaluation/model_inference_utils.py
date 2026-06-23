import os
import gc
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from argparse import ArgumentParser
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from glob import glob

n = 500

def perform_nemotron_stem_inference(llm, sampling_params, dataset_name="/home/ubuntu/AcquisitionSynthesis/data/nemotron_stem/test.parquet"):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:n]
    answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:n]

    prompt_template = lambda q: f"Answer the following multiple choice question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer (the letter of the answer choice) in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
    answers = [answer.split("\\boxed{")[-1].split("}")[0].strip() for answer in answers]
    
    return [{
        "experiment_name": "nemotron_stem",
        "task": "classification",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_nemotron_math_inference(llm, sampling_params, dataset_name="/home/ubuntu/AcquisitionSynthesis/data/nemotron_math/test.parquet"):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:n]
    answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:n]

    prompt_template = lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = llm.generate(prompts[:n], sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]
    answers = [answer.split("\\boxed{")[-1].split("}")[0].strip() for answer in answers]
    
    return [{
        "experiment_name": "nemotron_math",
        "task": "math",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_nemotron_chat_inference(llm, sampling_params, dataset_name="/home/ubuntu/AcquisitionSynthesis/data/nemotron_chat/test.parquet"):
    grounding_seed = pd.read_parquet(dataset_name, engine='pyarrow')
    questions = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))[:n]
    answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))[:n]

    prompt_template = lambda q: f"Answer the following question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<question> {q} </question>."
    prompts = [prompt_template(q) for q in questions]
    outputs = llm.generate(prompts[:n], sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]
    
    return [{
        "experiment_name": "nemotron_chat",
        "task": "open-ended",
        "questions": questions,
        "answers": answers,
        "outputs": outputs
    }]

def perform_mhotpot_inference(llm, sampling_params, data_dir="/home/ubuntu/lsk_mitigate/data/m_hotpotqa"):
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    experiments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[:n]
        queries = df['query']
        contexts = df['context']

        prompt_template = lambda c, q: f"Given some context, the task is the answer the question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer in <answer> </answer> tags.\n\n<context> {c} </context>\n<question> {q} </question>."
        prompts = [prompt_template(c, q) for c, q in zip(contexts, queries)]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]
    
        outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]
        questions = [f"Context: {c}\nQuestion: {q}" for c, q in zip(queries, contexts)]

        experiments.append({
            "experiment_name": f"mhotpot_{csv_file.split('/')[-1].split('.')[0]}",
            "task": "open-ended",
            "questions": questions,
            "answers": df['output'],
            "outputs": outputs
        })

    return experiments


def perform_mmmlu_inference(llm, sampling_params, data_dir="/home/ubuntu/AcquisitionSynthesis/data/mmmlu"):
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    experiments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[:n]
        questions = df['questions']

        prompt_template = lambda q: f"Answer the following multiple choice question. Output your reasoning in <reasoning> </reasoning> tags, and your final answer (the letter of the answer choice) in <answer> \\boxed{{}} </answer> tags.\n\n<question> {q} </question>."
        prompts = [prompt_template(q) for q in questions]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]
    
        outputs = [output.split("\\boxed{")[-1].split("}")[0].strip() for output in outputs]

        experiments.append({
            "experiment_name": f"mmmlu_{csv_file.split('/')[-1].split('.')[0]}",
            "task": "classification",
            "questions": questions,
            "answers": df['answers'],
            "outputs": outputs
        })

    return experiments

def perform_opus_inference(llm, sampling_params, data_dir="/home/ubuntu/AcquisitionSynthesis/data/opus-100"):
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")))
    experiments = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)[:n]
        questions = df['questions']

        prompt_template = lambda q: f"Answer the question and output your final answer in <answer> </answer> tags.\n<question> {q} </question>."
        prompts = [prompt_template(q) for q in questions]
        outputs = llm.generate(prompts, sampling_params=sampling_params)
        outputs = [output.outputs[0].text.strip() for output in outputs]
    
        outputs = [output.split("<answer>")[-1].split("</answer>")[0].strip() for output in outputs]

        experiments.append({
            "experiment_name": f"opus_{csv_file.split('/')[-1].split('.')[0]}",
            "task": "open-ended",
            "questions": questions,
            "answers": df['answers'],
            "outputs": outputs
        })

    return experiments
