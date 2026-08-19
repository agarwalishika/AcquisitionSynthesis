export CUDA_VISIBLE_DEVICES=0,1
py model_inference.py --model_name "Qwen/Qwen2.5-3B-Instruct"
py model_inference.py --model_name "Qwen/Qwen2.5-7B-Instruct"
py model_inference.py --model_name "meta-llama/Llama-3.1-8B-Instruct"
notify "hope all inferences are done"
