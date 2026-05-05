# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import datetime
from prompt_variations import *
import datasets
import json
from verl.utils.hdfs_io import copy, makedirs
import pyarrow as pa

def extract_last_boxed(s):
    try:
        temp = s.split("boxed")[-1]
        temp = temp[temp.index("{")+1:temp.rfind("}")]
        return temp
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="/home/ec2-user/grpo_synthesis/data/medmcqa/", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--train_size", type=int, default=500)
    parser.add_argument("--prompt", type=str, default="current")

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openlifescienceai/medmcqa"
    full_dataset = datasets.load_dataset(data_source)['train']
    full_dataset = full_dataset.map(lambda x: {'query': f"Question: {x['question']}\nA. {x['opa']}\nB. {x['opb']}\nC. {x['opc']}\nD. {x['opd']}"})
    names = full_dataset.features['cop'].names  # ['a', 'b', 'c', 'd']
    full_dataset = full_dataset.map(lambda x: {'answer': names[x['cop']]})
    full_dataset = full_dataset.map(lambda x: {'response': f"{x['answer'].upper()}. {x['op'+ x['answer']]}"})
    full_dataset = full_dataset.rename_column("exp", "reasoning")
    full_dataset = full_dataset.filter(lambda x: x['reasoning'] is not None)

    only_grounding = lambda q, a: f"""{{
    "question": "{q}",
    "answer": "{a}"
}}"""

    def create_dataset(ds, split):
        records = []
        for i in range(len(ds)):
            q = ds[i]['query']
            a = ds[i]['response']
            r = ds[i]['reasoning']
            records.append({
                "data_source": data_source + "_" + args.prompt,
                "prompt": [{
                    "role": "user",
                    "content": MEDICAL_PROMPT(q, a, r)
                }],
                "ability": "data_synthesis",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {
                        "split": split,
                        "index": len(records) + 1,
                        "question": only_grounding(q, a),
                        "grounding_question": q,
                        "grounding_answer": a,
                        "grounding_reasoning": a,
                }
            })

        return datasets.Dataset.from_list(records)
    
    def create_dataset_jsonl(ds, split):
        records = []
        
        for i in range(len(ds)):
            q = ds[i]['query']
            a = ds[i]['response']
            a = ds[i]['reasoning']

            records.append({
                "prompt": q,
                "completion": a,
                "id": len(records)
            })
            if len(records) >= 100:
                break
        return records


    TRAIN_SIZE = args.train_size
    TEST_SIZE = 5000
    VALID_SIZE = 100
    MAX_TOKENS = 5096

    # Use ~4 chars per token as a rough estimate to filter prompts
    def is_short_enough(example):
        q = example.get('query', '').strip().replace("\n", " ")
        a = example.get('response', '').strip().replace("\n", " ")
        r = example.get('reasoning', '').strip().replace("\n", " ")
        content = MEDICAL_PROMPT(q, a, r)
        return len(content) // 2 < MAX_TOKENS

    filtered = full_dataset.filter(is_short_enough)
    total_needed = (TRAIN_SIZE + TEST_SIZE + VALID_SIZE)
    assert len(filtered) >= total_needed, (
        f"Not enough examples after filtering: {len(filtered)} < {total_needed}"
    )

    train_dataset = create_dataset(filtered.select(range(TRAIN_SIZE)), "train")
    valid_dataset = create_dataset(filtered.select(range(TRAIN_SIZE, TRAIN_SIZE + VALID_SIZE)), "valid")
    test_dataset = create_dataset(filtered.select(range(len(filtered)-TEST_SIZE-1, len(filtered))), "test")
    all_dataset = create_dataset(filtered.select(range(0, len(filtered)-TEST_SIZE)), "all").select(range(0, 10000))

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    valid_dataset.to_parquet(os.path.join(local_save_dir, "valid.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    all_dataset.to_parquet(os.path.join(local_save_dir, "all.parquet"))

    # records = create_dataset_jsonl(filtered.select(range(TRAIN_SIZE)), "train")
    # for record in records:
    #     with open('/home/ec2-user/prismatic-synthesis/prismatic-synthesis/data/datasets/numina_seed.jsonl', 'a+') as f:
    #         f.write(json.dumps(record))
    #         f.write("\n")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)