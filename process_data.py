# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

"""
将 形如 {question, golden_answers, context} 的数据
预处理为 parquet (train + validation)，并映射成以下结构：
{
  "data_source": "hotpotqa/hotpot_qa",
  "prompt": [{"role": "user", "content": "Question: <q>\\n<instruction_following>"}],
  "ability": "multihop_qa",
  "reward_model": {"style": "rule", "ground_truth": ["答案1", "答案2", ...]}
}
"""

import os
import datasets
import argparse
import json
from verl.utils.hdfs_io import copy, makedirs  # HDFS 工具（公司内部库）

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./processed_2')
    parser.add_argument('--hdfs_dir', default=None)
    args = parser.parse_args()

    data_source = 'hotpotqa/hotpot_qa'
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # === 只保留 train 和 validation 文件 ===
    train_file = os.path.join(local_dir, "train.json")
    val_file   = os.path.join(local_dir, "validation.json")

    print("加载本地 JSON 文件...")
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(val_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    # ========== 新增：从 golden_answers 提取完整列表 ==========
    def extract_answers(item):
        """
        支持两种情况：
        1) 新数据：{"golden_answers": [...]} -> 返回整个 list（可能为空）
        2) 兼容旧数据：{"answer": "..."} -> 返回 [answer]
        其他情况统一返回 []
        """
        if 'golden_answers' in item:
            ga = item.get('golden_answers', [])
            if isinstance(ga, list):
                return [str(ans) for ans in ga]
            return []
        if 'answer' in item:
            return [str(item['answer'])]
        return []

    # 构建中间 dataset：只保留 question / answers 两列，供后续 map 使用
    train_dataset = datasets.Dataset.from_dict({
        'question': [item.get('question', '') for item in train_data],
        'answers':  [extract_answers(item)     for item in train_data],
    })

    validation_dataset = datasets.Dataset.from_dict({
        'question': [item.get('question', '') for item in validation_data],
        'answers':  [extract_answers(item)     for item in validation_data],
    })

    # 你的 instruction 提示，保持不变
    # instruction_following = (
    #     # r'You MUST FIRST think about the question, explain and analyze it, and rephrase it into a clear, explanatory statement, making it easier to understand so that large language models can interpret and answer it more accurately. '
    #     # r"You must first reflect on the question, interpret and analyze it so that the large language model can answer it more accurately. Then, based on the large model's response, you can reconsider and present the question again. Through repeated interactions with the large language model, you ultimately arrive at the answer."
    #     # r'The thinking process MUST BE enclosed within <think> </think> tags. '
    #     # r'The FINAL answer obtained by using the tool large language model must be placed in <answer> </answer> tags.'
        
    # )
#     instruction_following = (
#     "You should first think through the question, explaining and analyzing it. Then, give the question to the large language model to answer more accurately. Think about the large language model's response, and by interacting with the large language model again and again, arrive at the final answer. You must solve the task step by step with the following rules:\n"
#     "1. At the start and in each interaction with the large language model, write:\n"
#     "   <think>(your reasoning for this step)</think>\n"
#     "   <interaction_prompt>(the next action, idea, or request that moves the answer forward)</interaction_prompt>\n"
#     "2. Each <interaction_prompt> must build on what came before. Do not just repeat the same content. Let the content of the <interaction_prompt>...</interaction_prompt> evolve naturally (for example: outline → add details → refine → polish).\n"
#     "3. Continue producing think within <think> </think> and call tool within <interaction_prompt> </interaction_prompt> until the answer is ready.\n"
#     "4. When the solution is complete, write:\n"
#     "   <think>(your final reasoning)</think>\n"
#     "   <answer>(the final answer)</answer>"
# )

#     instruction_following = (
#         "First, provide a simple explanation of the question and give it to the large language model for a more accurate answer. Focus on explaining the question without deep reasoning in the first step. After receiving the response, think about the large language model's response, and by interacting with the large language model again and again, arrive at the final answer. Proceed step by step with the following rules:\n"
#         "1. Only in the first step, provide a brief explanation of the question and give it to the large language model for an answer:\n"
#         "   <think>(explain the question to the LLM and reason within 100 words and don't think deeply)</think>\n"
#         "   <interaction_prompt>(give the question and its explanation to the large language model)</interaction_prompt>\n"
#         "2. After the first step, in each interaction with the large language model, write:\n"
#         "   <think>(your reasoning for the receiving response and question)</think>\n"
#         "   <interaction_prompt>(new request to refine or validate the answer)</interaction_prompt>\n"
#         "3. Each <interaction_prompt> must build on what came before. Do not just repeat the same content. Let the content of the <interaction_prompt>...</interaction_prompt> evolve naturally (for example: outline → add details → refine → check). \n"
#         "4. Continue producing think within <think></think> and call tool within <interaction_prompt></interaction_prompt> until the answer is ready.\n"
#         "5. Once the answer is complete, write:\n"
#         "   <think>(final reasoning with the <interaction_response> and question)</think>\n"
#         "   <answer>(final answer for the question)</answer>"
# )
    instruction_following = (
        "First, provide a simple explanation of the question and give it to the large language model for a more accurate answer. Focus on explaining the question without deep reasoning in the first step. After receiving the response, think about the large language model's response, and by interacting with the large language model again and again, arrive at the final answer. Proceed step by step with the following rules:\n"
        "1. Only in the first step, provide a brief explanation of the question and give it to the large language model:\n"
        "   <think>(Brief thinking must not be over 80 words)</think>\n"
        "   <interaction_prompt>(give the question and its explanation to the large language model)</interaction_prompt>\n"
        "2. After the first step, in each interaction with the large language model, write:\n"
        "   <think>(your reasoning for the receiving response and question)</think>\n"
        "   <interaction_prompt>(new request to refine or validate the answer)</interaction_prompt>\n"
        "3. Each <interaction_prompt> must build on what came before. Do not just repeat the same content. Let the content of the <interaction_prompt>...</interaction_prompt> evolve naturally (for example: outline → add details → refine → check). \n"
        "4. Continue producing think within <think></think> and call tool within <interaction_prompt></interaction_prompt> until the answer is ready.\n"
        "5. Once the answer is complete, write:\n"
        "   <think>(final reasoning with the <interaction_response> and question)</think>\n"
        "   <answer>(final answer for the question)</answer>"
)

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question', '')
            answers_raw  = example.pop('answers', [])

            question = "Question: " + question_raw + '\n' + instruction_following

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "multihop_qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answers_raw   # 保留完整 list
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)

    # === 只保存 train + validation parquet ===
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
