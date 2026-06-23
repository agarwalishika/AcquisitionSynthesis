import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis/rewards/')
import requests
from format import parse
import os
SERVER_IP = os.environ['SERVER_IP']

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data = parse(solution_str)
    if data is None: return float(-1)
    
    repeat_penalty = compute_repeat_penalty(data['question']) 
    grounding_regularizer = compute_grounding_regularizer(data['question'])
    total_reward = (repeat_penalty + grounding_regularizer) / 2

    return total_reward

def compute_repeat_penalty(question):
    SERVER_A = f"http://{SERVER_IP}:5140/repeat_penalty"

    payload = {
        "question": question
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=300,
    )
    if not r.ok: return float(0.0)
    # r.raise_for_status()
    print("status:", r.status_code)
    print("body:", r.text)          # <- this will show {"detail":"..."}
    # print("json:", r.json())      # optional if it's JSON
    r.raise_for_status()

    repeat_penalty = r.json()["influence_rewards"]
    return repeat_penalty[0]

def compute_grounding_regularizer(question):
    SERVER_A = f"http://{SERVER_IP}:5140/grounding_regularizer"

    payload = {
        "question": question
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=300,
    )
    if not r.ok: return float(0.0)
    # r.raise_for_status()
    print("status:", r.status_code)
    print("body:", r.text)          # <- this will show {"detail":"..."}
    # print("json:", r.json())      # optional if it's JSON
    r.raise_for_status()

    repeat_penalty = r.json()["influence_rewards"]
    return repeat_penalty[0]