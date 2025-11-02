#!/bin/bash

export CHECKPOINT_DIR='checkpoints/Prompt-R1/Prompt-R1-qwen3-4b-gpt-4o-mini/global_step_4/actor'
export HF_MODEL_PATH='./Qwen/Qwen3-4B'
export TARGET_DIR='./merge_model/Prompt-R1_Qwen3-4B'

python3 verl/scripts/model_merger.py \
  --backend fsdp \
  --hf_model_path "$HF_MODEL_PATH" \
  --local_dir "$CHECKPOINT_DIR" \
  --target_dir "$TARGET_DIR"
