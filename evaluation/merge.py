import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned adapter/model")
args = parser.parse_args()

base_model_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_path = args.model_path
merged_path = args.model_path

model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved to {merged_path}")

# Push to Hugging Face Hub
model_name = args.model_path.split("/")[-1]
hf_repo_id = f"ishikauniphore/student_{model_name}"

model.push_to_hub(hf_repo_id)
tokenizer.push_to_hub(hf_repo_id)
print(f"Model pushed to https://huggingface.co/{hf_repo_id}")
