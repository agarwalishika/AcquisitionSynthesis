export CUDA_VISIBLE_DEVICES=0,1,2,3
cd evaluation

py sft.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/PS_qwen3bins_medmcqa.parquet"
rm -rf /tmp/sft_models/PS_qwen3bins_medmcqa
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_PS_qwen3bins_medmcqa" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"

py sft.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/PS_qwen3bins_numina.parquet"
rm -rf /tmp/sft_models/PS_qwen3bins_numina
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_PS_qwen3bins_numina" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"
