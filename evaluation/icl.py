import os
from vllm import LLM, SamplingParams
from argparse import ArgumentParser
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from transformers import AutoTokenizer


def find_closest_examples(prompts, train_dataset):
    embedder = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
    train_questions = list(train_dataset['question'])

    train_embs = embedder.encode(train_questions, convert_to_tensor=True, normalize_embeddings=True)
    prompt_embs = embedder.encode(prompts, convert_to_tensor=True, normalize_embeddings=True)

    # shape: (n_prompts, n_train) — pick argmax per row
    best_indices = (prompt_embs @ train_embs.T).argmax(dim=1).cpu().tolist()

    del embedder
    torch.cuda.empty_cache()

    return [
        {
            'question': train_dataset.iloc[i]['question'],
            'reasoning': train_dataset.iloc[i]['reasoning'],
            'answer': train_dataset.iloc[i]['answer'],
        }
        for i in best_indices
    ]


def apply_chat_template(model_name, prompts, icl_examples=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    results = []
    for i, p in enumerate(prompts):
        messages = []
        if icl_examples is not None:
            ex = icl_examples[i]
            messages.append({"role": "user", "content": f"Answer the following question. Output your answer in <reasoning> </reasoning> and <answer> </answer> tags.\n<question> {ex['question']} </question>"})
            messages.append({"role": "assistant", "content": f"<reasoning> {ex['reasoning']} </reasoning>\n<answer> {ex['answer']} </answer>"})
        messages.append({"role": "user", "content": f"Answer the following question. Output your answer in <reasoning> </reasoning> and <answer> </answer> tags.\n<question> {p} </question>"})
        results.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        ))
    return results


def generate_outputs(args, train_dataset, prompts):
    icl_examples = find_closest_examples(prompts, train_dataset) if train_dataset is not None else None
    prompts = apply_chat_template(args.model_name, prompts, icl_examples)

    llm = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    outputs = llm.generate(prompts, sampling_params=sampling_params)
    outputs = [output.outputs[0].text.strip() for output in outputs]

    outputs = [output.split("<answer>")[-1].replace("</answer>", "").strip() for output in outputs]

    del llm, sampling_params
    return outputs

def evaluate_outputs(predictions, references):
    embedding_model = LLM('Qwen/Qwen3-Embedding-0.6B', task="embed")

    pred_embs = embedding_model.embed(predictions)
    pred_embs = torch.tensor([o.outputs.embedding for o in pred_embs])

    ref_embs = embedding_model.embed(references)
    ref_embs = torch.tensor([o.outputs.embedding for o in ref_embs])

    scores = (pred_embs @ ref_embs.T).diag()

    return scores

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="/tmp/sft_models/random_OpenR1-Math-220k_100")
    parser.add_argument("--train_dataset_name", default=None)
    parser.add_argument("--test_dataset_name", default="/home/ec2-user/grpo_synthesis/data/numina/test.parquet")
    parser.add_argument("--use_chat_template", action="store_true",
                        help="Apply chat template to prompts (use for SFT-trained models)")
    args = parser.parse_args()

    train_dataset = pd.read_parquet(args.train_dataset_name, engine='pyarrow')

    grounding_seed = pd.read_parquet(args.test_dataset_name, engine='pyarrow')
    test_prompts = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_question'], axis=1))
    test_answers = list(grounding_seed.apply(lambda row: row['extra_info']['grounding_answer'], axis=1))

    outputs = generate_outputs(args, train_dataset, test_prompts)

    scores = evaluate_outputs(outputs, test_answers)

    exp_name = f"{args.train_dataset_name.split('/')[-1].split('.')[0]}{args.test_dataset_name.split('/')[-2]}"
    prompts = [p.replace("\n", " ") for p in test_prompts]
    answers = [a.replace("\n", " ") for a in test_answers]
    outputs = [o.replace("\n", " ") for o in outputs]
    pd.DataFrame.from_dict({
        "question": prompts,
        "predictions": outputs,
        "references": answers,
        "scores": scores
    }).to_csv(f"{exp_name}.csv", sep="|")

    with open('results.txt', 'a+') as f:
        f.write(f"ICL_{exp_name}\t{float((scores > 0.8).sum() / len(scores))}\n")
