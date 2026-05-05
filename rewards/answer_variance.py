import sys
sys.path.append('/home/ec2-user/grpo_synthesis/rewards/')
import requests
from format import parse
import os
import torch

SERVER_IP = os.environ['SERVER_IP']

def compute_answer_variance_batch(data, kwargs):
    from vllm import LLM, SamplingParams
    import torch, hdbscan

    model_name = kwargs["model_name"]
    language_model = LLM(model_name, tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True, max_model_len=4096)
    llm_sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    prompts = []
    for d in data:
        for _ in range(16): 
            prompts.append(d)

    outputs = language_model.generate(prompts, sampling_params=llm_sampling_params)
    outputs = [o.outputs[0].text.strip() for o in outputs]

    del language_model, llm_sampling_params
    torch.cuda.empty_cache()

    embedding_model = LLM("Qwen/Qwen3-Embedding-0.6B", task="embed")
    rewards = []
    for i in range(0, len(outputs), 16):
        answer_embeddings = embedding_model.embed(outputs[i:i+16])
        answer_embeddings = torch.Tensor([a.outputs.embedding for a in answer_embeddings])

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
        labels = clusterer.fit_predict(answer_embeddings)
        rewards.append(len(set(labels)) / 16.0)
    
    return rewards


def compute_answer_variance(data):
    SERVER_A = f"http://{SERVER_IP}:5145/answer_variance"
    payload = {
        "data": data,
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=3000,
    )
    if not r.ok: return float(0.0)
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    av_reward = float(r.json()["acquisition_reward"]) / 16.0

    return av_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)
    av_rewards = compute_answer_variance(data)

    return av_rewards + xml_reward