export CUDA_VISIBLE_DEVICES=1,2,6,7
python -m vllm.entrypoints.openai.api_server \
  --model gpt-oss-20b \
  --port 8006 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.70 \
#   --dtype bfloat16 \
#   --max-model-len 32768 \
#   --max-num-seqs 32