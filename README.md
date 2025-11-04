# Prompt-R1: Enhancing LLM interaction on behalf of humans

Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning [[paper](https://arxiv.org/abs/2511.01016)]


## Overview

<div align="center">
  <img src="figs/fig1.png" width="80%"/>
</div>

**Prompt-R1** has addressed a critical challenge in interacting with large language models (LLMs)—the inability of users to provide accurate and effective interaction prompts for complex tasks. **Prompt-R1** is an **end-to-end reinforcement learning (RL)** framework that enhances the performance of LLMs by facilitating **collaborative automatic prompting** between a small-scale LLM and a large-scale LLM. **Prompt-R1**, through **multi-turn prompt interaction**, significantly improves the generation quality and reasoning accuracy of large-scale LLMs, enabling better task-solving performance without requiring user expertise in prompt formulation.



<div align="center">
  <img src="figs/fig2.png" width="90%"/>
</div>

By integrating **collaborative prompting** and **reinforcement learning**, **Prompt-R1** offers a **plug-and-play framework** that supports both **inference** and **training** with **various large-scale LLMs** as the environment. 

## Experimental Results
**Results of Different Large language models:**
<div align="center">
  <img src="figs/fig3.png" width="100%"/>
</div>



## Prompt-R1 Implementation

### Install Environment
```bash
conda create -n promptr1 python==3.12 -y
conda activate promptr1
cd verl
pip3 install -e .
pip3 install vllm==0.8.3
pip3 install flash-attn==2.7.4.post1  # Download: https://github.com/Dao-AILab/flash-attention/releases
pip3 install FlagEmbedding faiss-cpu
pip3 install debugpy==1.8.0 "ray[default]" debugpy
```

### Dataset Preparation
>Our datasets are in:
```bash
Training Dataset: dataset\train_data
Evaluation Dataset: dataset\eval_data
```

### Quick Start: Prompt-R1 


### 1. To use closed source LLM, modify promptr1_agent\tool\tools\LLM-toolpy:
```bash
API_KEY = "your_api_key"
MODEL = "model_name"
BASE_URL = "url"
```
>Run:
```bash
nohup bash run_prompt-R1.sh > Prompt-R1_training.out &
```


### 2. Deploy an Open-Source Model Locally
#### 1. Install vLLM and dependencies
```bash
# Create environment
conda create -n vllmapi python=3.12 -y
conda activate vllmapi
# Install dependencies
pip3 install transformers accelerate huggingface_hub
pip3 install vllm
```

#### 2. Start the OpenAI-compatible server:
```bash
nohup bash vllm_api.sh > api.out 2>&1 &
```
#### 3. To use closed source LLM, modify promptr1_agent\tool\tools\LLM-toolpy to call your local API:
>Edit agent_r1/tool/tools/search_tool.py and set the local API endpoint and model name
```bash
base_url = "http://<SERVER_IP>:8006/v1"
```

### Evaluation
#### 1.Edit model_merge.sh and set the paths:
```bash
export CHECKPOINT_DIR='checkpoints/Prompt-R1/Prompt-R1-qwen3-4b-gpt-4o-mini/global_step_320/actor'
export HF_MODEL_PATH='./Qwen/Qwen3-4B'
export TARGET_DIR='./merge_model/Prompt-R1_Qwen3-4B'
```

#### 2.Edit vllm_serve.sh:
```bash
export MODEL_NAME='./merge_model/Prompt-R1_Qwen3-4B'
```

#### 3.Inference
```bash
python inference.py
```

#### 4.Batch inference & Evaluation
```bash
python batch_inference.py
python eval_scores.py
```

## BibTex

If you find this work is helpful for your research, please cite:

```bibtex
@misc{liu2025promptr1collaborativeautomaticprompting,
      title={Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning}, 
      author={Wenjin Liu and Haoran Luo and Xueyuan Lin and Haoming Liu and Tiesunlong Shen and Jiapu Wang and Rui Mao and Erik Cambria},
      year={2025},
      eprint={2511.01016},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.01016}, 
}
```

For further questions, please contact: wenjinliu23@outlook.com.

## Acknowledgement

This repo benefits from [Agent-R1](https://github.com/0russwest0/Agent-R1), [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher), [Graph-R1](https://github.com/LHRLAB/Graph-R1), and [Search-R1](https://github.com/RUCAIBox/R1-Searcher). Thanks for their wonderful works.