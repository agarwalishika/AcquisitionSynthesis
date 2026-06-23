### STEP 2: Dataset generations
export CUDA_VISIBLE_DEVICES=6,7
py generating_data/data_gen_cluster.py \
    --dataset_name "numina" \
    --acquisition_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/generating_data/training_data/base_qwen7bins_numina.parquet" \
    --size 1000 --k 16

py generating_data/data_gen_cluster.py \
    --dataset_name "medmcqa" \
    --acquisition_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --answer_model_name "Qwen/Qwen2.5-7B-Instruct" \
    --output_file "/home/ubuntu/AcquisitionSynthesis/generating_data/training_data/base_qwen7bins_medmcqa.parquet" \
    --size 1000 --k 16

# ### STEP 3: Student evaluation
cd evaluation
py sft.py \
    --file_name "/home/ubuntu/AcquisitionSynthesis/generating_data/training_data/base_qwen7bins_numina.parquet" \
    --model_name "Qwen/Qwen2.5-7B-Instruct"
rm -rf /tmp/sft_models/base_qwen7bins_numina

py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_base_qwen7bins_numina" \
    --dataset "/home/ubuntu/AcquisitionSynthesis/data/numina/test.parquet"

py sft.py \
    --file_name "/home/ubuntu/AcquisitionSynthesis/generating_data/training_data/base_qwen7bins_medmcqa.parquet" \
    --model_name "Qwen/Qwen2.5-7B-Instruct"
rm -rf /tmp/sft_models/base_qwen7bins_medmcqa

py model_inference.py \
    --model_name "{HF_USERNAME}/acquisition_student_base_qwen7bins_medmcqa" \
    --dataset "/home/ubuntu/AcquisitionSynthesis/data/medmcqa/test.parquet"
cd ..