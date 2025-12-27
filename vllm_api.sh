export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m vllm.entrypoints.openai.api_server \
  --model gpt-oss-20b \
  --port 8006 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.70 \
#   --dtype bfloat16 \
#   --max-model-len 32768 \
#   --max-num-seqs 32