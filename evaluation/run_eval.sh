MODEL_NAMES=("Qwen/Qwen2.5-7B-Instruct" "ishikauniphore/student_3bT-7bS-v2_nemotron_stem_mcot" "ishikauniphore/student_3bT-7bS-v1_nemotron_stem_mcot")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    py model_inference.py --model ${MODEL_NAME}
    py model_embed.py --model ${MODEL_NAME}
    py model_rouge.py --model ${MODEL_NAME}
    py model_laj.py --model ${MODEL_NAME}
    notify "full evals for ${MODEL_NAME}"
done