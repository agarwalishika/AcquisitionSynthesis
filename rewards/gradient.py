import sys
sys.path.append('/home/ec2-user/grpo_synthesis/rewards/')
from format import parse
from repeat_penalty import compute_repeat_penalty
import requests
import numpy as np

import os
SERVER_IP = os.environ['SERVER_IP']

def compute_gradient(data):
    SERVER_A = f"http://{SERVER_IP}:5145/gradient"

    payload = {
        "data": data,
    }

    for _ in range(3):
        r = requests.post(
            SERVER_A,
            json=payload,
            headers={"X-API-Key": ""},
            timeout=300,
        )
        if r.ok: break
    if not r.ok: return float(0.0)
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    gradient_mag = r.json()["acquisition_reward"]

    return gradient_mag

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    data, xml_reward = parse(solution_str)
    if data is None: return float(0.0)

    gradient_mag = compute_gradient(data)
    return gradient_mag + xml_reward