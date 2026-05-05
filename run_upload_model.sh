LATEST_CKPT=$(ls -dt /tmp/grpo_synthesis_models/$1/global_step_* | head -1)

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $LATEST_CKPT/actor \
    --target_dir /tmp/grpo_synthesis_models/$1

py upload_to_hf.py --model_name "/tmp/grpo_synthesis_models/$1" --shorthand "acquisition_$1"