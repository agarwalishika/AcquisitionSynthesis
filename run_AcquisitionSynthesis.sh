REWARD=$1
DATASET=$2

### STEP 1: Acquisition training
rm -rf /tmp/grpo_synthesis_models
GRPO_KWARGS="{\"model_name\": \"Qwen/Qwen2.5-3B-Instruct\", \"dataset_name\": \"/home/ec2-user/grpo_synthesis/data/${DATASET}/train.parquet\"}"
source run_verl.sh "Qwen/Qwen2.5-3B-Instruct" "rewards/${REWARD}.py" "qwen3bins_${DATASET}_${REWARD}" "${DATASET}" "${REWARD}" "$GRPO_KWARGS"

### STEP 2: Dataset generations
export CUDA_VISIBLE_DEVICES=0,1,2,3
py generating_data/data_gen_cluster.py \
    --dataset_name "${DATASET}" \
    --acquisition_model_name "{HF_USERNAME}/acquisition_qwen3bins_${DATASET}_${REWARD}" \
    --answer_model_name "Qwen/Qwen2.5-3B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/qwen3bins_${DATASET}_${REWARD}.parquet" \
    --size 1000 --k 16

### STEP 3: Student evaluation
cd evaluation
rm -rf /tmp/sft_models/
py sft.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/qwen3bins_${DATASET}_${REWARD}.parquet"

py model_inference.py \
    --model_name "/tmp/sft_models/qwen3bins_${DATASET}_${REWARD}" \
    --dataset "/home/ec2-user/grpo_synthesis/data/numina/test.parquet"
py model_inference.py \
    --model_name "/tmp/sft_models/qwen3bins_${DATASET}_${REWARD}" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"
py model_inference.py \
    --model_name "/tmp/sft_models/qwen3bins_${DATASET}_${REWARD}" \
    --dataset "/home/ec2-user/grpo_synthesis/data/aime/test.parquet"
py model_inference.py \
    --model_name "/tmp/sft_models/qwen3bins_${DATASET}_${REWARD}" \
    --dataset "/home/ec2-user/grpo_synthesis/data/pubmedqa/test.parquet"
py model_inference.py \
    --model_name "/tmp/sft_models/qwen3bins_${DATASET}_${REWARD}" \
    --dataset "/home/ec2-user/grpo_synthesis/data/codeforces/test.parquet"

cd ..