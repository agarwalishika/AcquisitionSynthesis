########################################
###             RANDOM               ###
########################################

### STEP 1: select data
py generating_data/baseline_selection_random.py --data "numina"
py generating_data/baseline_selection_random.py --data "medmcqa"

### STEP 2: Dataset generations
export CUDA_VISIBLE_DEVICES=0,1,2,3
py generating_data/data_gen_cluster.py \
    --dataset_name "numina" \
    --questions_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/random_numina.parquet" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/random_numina.parquet" \
    --size 1000 --k 16

py generating_data/data_gen_cluster.py \
    --dataset_name "medmcqa" \
    --questions_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/random_medmcqa.parquet" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/random_medmcqa.parquet" \
    --size 1000 --k 16

### STEP 3: Student evaluation
cd evaluation

py sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/random_numina.parquet"
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_random_numina" \
    --dataset "/home/ec2-user/grpo_synthesis/data/numina/test.parquet"
rm -rf /tmp/sft_models/random_numina

py sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/random_medmcqa.parquet"
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_random_medmcqa" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"
rm -rf /tmp/sft_models/random_medmcqa

cd ..







########################################
###            FILTERED              ###
########################################

### Step 1: Filter and create data
py baseline_selection_filtered.py \
    --data "numina" --model "Qwen/Qwen2.5-7B-Instruct" \
    --dir_name "qwen7b_numina" --size 1000 \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_numina.parquet"

export CUDA_VISIBLE_DEVICES=0,1,2,3
py baseline_selection_filtered.py \
    --data "medmcqa" --model "Qwen/Qwen2.5-7B-Instruct" \
    --dir_name "qwen7b_medmcqa" --size 1000 \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_medmcqa.parquet"

### STEP 2: Dataset generations
cd generating_data
py data_gen_cluster.py \
    --dataset_name "numina" \
    --questions_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_numina.parquet" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_numina.parquet" \
    --size 1000 --k 16

py data_gen_cluster.py \
    --dataset_name "medmcqa" \
    --questions_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_medmcqa.parquet" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_medmcqa.parquet" \
    --size 1000 --k 16

### STEP 3: Student evaluation
cd evaluation

py sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_numina.parquet"
rm -rf /tmp/sft_models/filtered_qwen7bins_numina
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_filtered_qwen7bins_numina" \
    --dataset "/home/ec2-user/grpo_synthesis/data/numina/test.parquet"



py sft.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --file_name "/home/ec2-user/grpo_synthesis/generating_data/training_data/filtered_qwen7bins_medmcqa.parquet"
rm -rf /tmp/sft_models/filtered_qwen7bins_medmcqa
py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_filtered_qwen7bins_medmcqa" \
    --dataset "/home/ec2-user/grpo_synthesis/data/medmcqa/test.parquet"