import os
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig

def format_prompt(example, tokenizer):
    messages = [
        {"role": "user", "content": f"Answer the following question. Output your answer in <reasoning> </reasoning> and <answer> </answer> tags.\n<question> {example['question']} </question>"},
        {"role": "assistant", "content": f"<reasoning> {example['reasoning']} </reasoning>\n<answer> {example['answer']} </answer>"},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def sft_train(file_name, model_name, num_epochs=5, output_dir="/tmp/sft_models/", student_name=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_parquet(file_name)[:1000]
    assert "question" in df.columns and "answer" in df.columns and "reasoning" in df.columns, \
        "CSV must have 'question' and 'answer' columns"

    dataset = Dataset.from_pandas(df[["question", "answer", "reasoning"]].dropna())
    dataset = dataset.map(lambda ex: format_prompt(ex, tokenizer))

    run_name = file_name.split('/')[-1].split('.')[0]
    save_path = os.path.join(output_dir, run_name)
    os.makedirs(save_path, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        dataset_text_field="text",
        report_to="none",
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    HF_USERNAME = os.getenv("HF_USERNAME")
    if student_name is None:
        student_name = save_path.split("/")[-1]
    model.push_to_hub(f"{HF_USERNAME}/acquisition_student_{student_name}")
    tokenizer.push_to_hub(f"{HF_USERNAME}/acquisition_student_{student_name}")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", help="Path to CSV file with 'question' and 'answer' columns", default="/home/ec2-user/grpo_synthesis/generating_data/ins_train/random_OpenR1-Math-220k_1000.parquet")
    parser.add_argument("--model_name",  default="Qwen/Qwen2.5-3b-Instruct")
    parser.add_argument("--output_dir", default="/tmp/sft_models/", help="Directory to save trained model")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--student_name", type=str, default=None)
    args = parser.parse_args()

    model_path = sft_train(args.file_name, args.model_name, args.num_epochs, args.output_dir, args.student_name)