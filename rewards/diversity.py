import sys
sys.path.append('/home/ec2-user/grpo_synthesis/rewards/')
import requests
from format import parse
from repeat_penalty import compute_score as compute_regularizing_score
import os

SERVER_IP = os.environ['SERVER_IP']

def compute_diversity(data):
    SERVER_A = f"http://{SERVER_IP}:5145/diversity"
    payload = {
        "data": data,
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=30,
    )
    if not r.ok: return float(0.0)
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    diversity_reward = min(max(r.json()["acquisition_reward"], 0.0), 2.0)

    return diversity_reward


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)
    diversity_rewards = compute_diversity(data)

    return diversity_rewards + xml_reward