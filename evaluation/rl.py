import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import re
from trl import GRPOConfig, GRPOTrainer

HF_USERNAME = os.getenv("HF_USERNAME")

def compute_score(prompts, completions, completion_ids, **kwargs):
    rewards = []
    for i in range(len(completions)):
        temp = completions[i][0]['content']
        if "<answer>" not in temp:
            rewards.append(-1.0)
        else:
            rewards.append(
                kwargs['solution'][i] in temp.split("<answer>")[-1].split("</answer")[0]
            )
        
    return rewards

def extract_final_answer(text):
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1) if match else None

def format_prompt(example, tokenizer):
    # _pdb()
    messages = [
        {"role": "user", "content": f"Answer the following question. Output your answer in <reasoning> </reasoning> and <answer> </answer> tags.\n<question> {example['question']} </question>"}
    ]

    return {"prompt": messages, "solution": example["answer"]}

def _pdb():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        import pdb; pdb.set_trace()

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)


def rl_train(file_name, model_name, output_dir):
    rank0_print("\n ----- USING GRPO TO TRAIN MODEL")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    df = pd.read_parquet(file_name)[:1000]
    assert "question" in df.columns and "answer" in df.columns and "reasoning" in df.columns, \
        "CSV must have 'question' and 'answer' columns"

    dataset = Dataset.from_pandas(df[["question", "answer", "reasoning"]].dropna())
    dataset = dataset.map(lambda ex: format_prompt(ex, tokenizer))

    # _pdb()
    # dataset = Dataset.from_pandas(df[["question", "answer", "reasoning"]].dropna()) # 1000
    # dataset = dataset.map(lambda ex: format_prompt(ex, tokenizer))
    # dataset = dataset.filter(lambda ex: ex["solution"] is not None) # 751
    # dataset = dataset.remove_columns(['question', 'answer', 'reasoning'])

    run_name = file_name.split('/')[-1].split('.')[0]
    save_path = os.path.join(output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    
    grpo_config = GRPOConfig(
        output_dir=save_path,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=24,
        num_generations=4,  # Specific to GRPO
        beta=0.1,          # KL coefficient
        learning_rate=1e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        max_completion_length=256,
        gradient_checkpointing=True,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=compute_score, 
        train_dataset=dataset,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    model.push_to_hub(f"{HF_USERNAME}/acquisition_student_RL_{run_name}")
    tokenizer.push_to_hub(f"{HF_USERNAME}/acquisition_student_RL_{run_name}")

    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default= "/home/ec2-user/grpo_synthesis/generating_data/training_data/qwen3bins_numina_confidence.parquet", help="Path to CSV file with 'question' and 'answer' columns")
    parser.add_argument("--student_model", default="meta-llama/Llama-3.1-8B-Instruct", help="Directory to save trained model")
    parser.add_argument("--output_dir", default="/tmp/sft_models/", help="Directory to save trained model")
    args = parser.parse_args()

    model_path = rl_train(args.file_name, args.student_model, args.output_dir)