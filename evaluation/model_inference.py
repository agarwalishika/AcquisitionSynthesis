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
from model_inference_utils import *
import pickle

# def evaluate_embed_sim(questions, answers, outputs, embedding_model):
#     pred_embs = embedding_model.embed(outputs)
#     pred_embs = torch.tensor([o.outputs.embedding for o in pred_embs])

#     ref_embs = embedding_model.embed(answers)
#     ref_embs = torch.tensor([o.outputs.embedding for o in ref_embs])

#     embed_sim = (pred_embs @ ref_embs.T).diag()

#     return embed_sim



# def evaluate_rouge(questions, answers, outputs, rouge_scorer):
#     return [rouge_scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(outputs, answers)]



# PROMETHEUS_PROMPT = lambda q, a, o: f"""###Task Description:
# A question and a response will be given. Evaluate the correctness and accuracy of the response based on the reference answer. You MUST give a score between 1 and 5. Respond strictly with only: [RESULT] (score)

# ###Question:
# {q}

# ###Response to Evaluate:
# {a}

# ###Reference Answer (Score 5):
# {o}

# ###Score Rubrics:
# [Is the response correct and accurate?]
# Score 1: The response is completely incorrect or irrelevant.
# Score 2: The response is mostly incorrect with minor correct elements.
# Score 3: The response is partially correct but lacks accuracy or completeness.
# Score 4: The response is mostly correct with minor errors.
# Score 5: The response is completely correct and accurate.

# ###Feedback:"""
# def evaluate_laj(questions, answers, outputs, llm, sampling_params):

#     judge_prompts = [PROMETHEUS_PROMPT(q, a, o) for q, a, o in zip(questions, answers, outputs)]
#     judge_outputs = llm.generate(judge_prompts, sampling_params)
#     judge_outputs = [o.outputs[0].text.strip() for o in judge_outputs]

#     def parse_score(text):
#         match = re.search(r'\[RESULT\]\s*(\d)', text)
#         return int(match.group(1)) if match else None

#     judge_scores = [parse_score(o) for o in judge_outputs]

#     return judge_scores

# def clear(model):
#     if hasattr(model, 'llm_engine') and hasattr(model.llm_engine, 'model_executor'):
#         try:
#             model.llm_engine.model_executor.shutdown()
#         except Exception:
#             pass
#     destroy_model_parallel()
#     destroy_distributed_environment()
#     for key in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE']:
#         os.environ.pop(key, None)
#     del model
#     gc.collect()
#     torch.cuda.synchronize()
#     torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="ishikauniphore/student_3bT-7bS-v2_nemotron_stem_mcot")
    args = parser.parse_args()

    ######## LLM INFERENCE ########
    llm = LLM(args.model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
    experiments = []
    experiments.extend(perform_opus_inference(llm, sampling_params))
    experiments.extend(perform_mmmlu_inference(llm, sampling_params))
    experiments.extend(perform_mhotpot_inference(llm, sampling_params))
    experiments.extend(perform_nemotron_stem_inference(llm, sampling_params))
    experiments.extend(perform_nemotron_math_inference(llm, sampling_params))
    experiments.extend(perform_nemotron_chat_inference(llm, sampling_params))


    with open(f"eval_{args.model_name.split('/')[-1]}.pkl", 'wb+') as f:
        pickle.dump(experiments, f)

    # clear(llm)
    # llm = None
    # gc.collect()
    # torch.cuda.empty_cache()
    # ######## LLM INFERENCE ########



    # ######## EMBED SIMILARITY ########
    # embedding_model = LLM('Qwen/Qwen3-Embedding-8B', task="embed", tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.5)
    # for exp in experiments:
    #     exp['embed_sim'] = evaluate_embed_sim(exp['questions'], exp['answers'], exp['outputs'], embedding_model)
    # clear(embedding_model)
    # embedding_model = None
    # gc.collect()
    # torch.cuda.empty_cache()
    # ######## EMBED SIMILARITY ########


    # ######## ROUGE-L ########
    # rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # for exp in experiments:
    #     exp['rouge_l'] = evaluate_rouge(exp['questions'], exp['answers'], exp['outputs'], rouge_scorer)
    # del rouge_scorer
    # ######## ROUGE-L ########


    # ######## LLM-AS-A-JUDGE ########
    # judge_model_name = 'prometheus-eval/prometheus-7b-v2.0'
    # judge_llm = LLM(judge_model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7)
    # judge_params = SamplingParams(temperature=0.0, max_tokens=256)
    # for exp in experiments:
    #     exp['judge_score'] = evaluate_laj(exp['questions'], exp['answers'], exp['outputs'], judge_llm, judge_params)
    # clear(judge_llm)
    # ######## LLM-AS-A-JUDGE ########



    ######## RECORDING EVALUATION ########
    # for exp in experiments:
    #     exp_name = f"{args.model_name.split('/')[-1]}_{exp['experiment_name']}"
    #     prompts = [p.replace("\n", " ") for p in exp['questions']]
    #     answers = [a.replace("\n", " ") for a in exp['answers']]
    #     outputs = [o.replace("\n", " ") for o in exp['outputs']]
    #     pd.DataFrame.from_dict({
    #         "question": prompts,
    #         "predictions": outputs,
    #         "references": answers,
    #         # "embed_sim": exp['embed_sim'],
    #         # "rouge_l": exp['rouge_l'],
    #         # "judge_score": exp['judge_score'],
    #     }).to_csv(f"{exp_name}.csv", sep="|")

        # with open('results.txt', 'a+') as f:
        #     f.write(f"{exp_name}\tembed_sim={float((exp['embed_sim'] > 0.7).sum() / len(exp['embed_sim'])):.3f}\trouge_l={sum(s >= 0.5 for s in exp['rouge_l']) / len(exp['rouge_l']):.3f}\tjudge={sum(s >= 4 for s in exp['judge_score']) / len(exp['judge_score']):.3f}\n")
    ######## RECORDING EVALUATION ########
