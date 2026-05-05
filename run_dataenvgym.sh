export CUDA_VISIBLE_DEVICES=6,7
cd generating_data
py baseline_synthesis_DataEnvGym.py \
    --data "numina" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/DataEnvGym_numina_qwen7bins.parquet"

py baseline_synthesis_DataEnvGym.py \
    --data "medmcqa" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/DataEnvGym_medmcqa_qwen7bins.parquet"

cd ..
cd evaluation

py sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/DataEnvGym_medmcqa_qwen7bins.parquet"
rm -rf /tmp/sft_models/DataEnvGym_medmcqa_qwen7bins
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_DataEnvGym_medmcqa_qwen7bins" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_DataEnvGym_medmcqa_qwen7bins" \
    --dataset "/home/ec2-user/grpo_synthesis/data/numina/test.parquet"


py sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/DataEnvGym_numina_qwen7bins.parquet"
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_DataEnvGym_numina_qwen7bins" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_DataEnvGym_numina_qwen7bins" \
    --dataset "/home/ec2-user/grpo_synthesis/data/numina/test.parquet"
rm -rf /tmp/sft_models/DataEnvGym_numina_qwen7bins
