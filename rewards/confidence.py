import sys
sys.path.append('/home/ec2-user/grpo_synthesis/rewards/')
from format import parse
from repeat_penalty import compute_repeat_penalty, compute_grounding_regularizer
import requests
import torch
import os
SERVER_IP = os.environ['SERVER_IP']

def compute_confidence_batch(data, kwargs):
    from vllm import LLM, SamplingParams
    model_name = kwargs['model_name']
    language_model = LLM(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.7, trust_remote_code=True, max_model_len=4096)
    llm_sampling_params = SamplingParams(temperature=0.7, logprobs=2, max_tokens=512)
    outputs = language_model.generate(data, llm_sampling_params)

    rewards = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs
        avg_diff = 0
        for logprob in logprobs:
            keys = list(logprob.keys())
            try:
                avg_diff += (torch.exp(torch.Tensor([logprob[keys[0]].logprob])) -
                            torch.exp(torch.Tensor([logprob[keys[1]].logprob])))
            except Exception:
                continue
        avg_diff /= len(logprobs)
        rewards.append(float(1.0/avg_diff))

    return rewards

def compute_confidence(data):
    SERVER_A = f"http://{SERVER_IP}:5145/confidence"
    payload = {
        "data": data,
    }

    for _ in range(3):
        r = requests.post(
            SERVER_A,
            json=payload,
            headers={"X-API-Key": ""},
            timeout=600,
        )
        if r.ok: break
    if not r.ok: return float(0.0)
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    if r.json()["acquisition_reward"] is not None:
        conf_reward = min(max(r.json()["acquisition_reward"], 0.0), 2.0)
    else:
        return float(0.0)
    return conf_reward

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)

    conf_reward = compute_confidence(data)
    return conf_reward + xml_reward