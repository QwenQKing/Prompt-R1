# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random
# from eval import cal_em

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def cal_f1_score(prediction, ground_truth):
    """
    Compute F1 score between prediction and ground truth.
    ground_truth can be a single string or a list of strings.
    Returns the maximum F1 score among all ground truth answers.
    """
    # Convert ground_truth to list if it's a single string
    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth
    
    max_f1 = 0.0
    pred_tokens = normalize_answer(prediction).split()
    
    for gt in ground_truth_list:
        gold_tokens = normalize_answer(gt).split()
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        
        max_f1 = max(max_f1, f1)
    
    return max_f1

def exact_match_score(prediction, ground_truth):
    """
    Compute Exact Match (EM) between prediction and ground truth.
    ground_truth can be a single string or a list of strings.
    Returns 1.0 if prediction matches any ground truth answer, 0.0 otherwise.
    """
    # Convert ground_truth to list if it's a single string
    if isinstance(ground_truth, str):
        ground_truth_list = [ground_truth]
    else:
        ground_truth_list = ground_truth
    
    normalized_prediction = normalize_answer(prediction)
    
    for gt in ground_truth_list:
        if normalize_answer(gt) == normalized_prediction:
            return 1.0
    
    return 0.0

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    em_score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            em_score = 1.0
            break
    return em_score

def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    sub_score = 0.0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            sub_score = 1.0
            break
    return sub_score

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def has_answer_tags(solution_str):
    """Check if solution contains <answer> tags with non-empty content."""
    if solution_str is None:
        return False
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, solution_str, re.DOTALL)
    if match and match.group(1).strip():
        return True
    return False

# def compute_score_format(solution_str):
#     """The scoring function for format reward - 修改版本，增加answer标签奖励。
#     Args:
#         solution_str: the solution text
#     """
#     if solution_str is None:
#         return 0.0
#     try:
#         assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
#         format_reward = 0.0
#         # If no blocks found, return 0
#         if not assistant_blocks or len(assistant_blocks) == 0:
#             return 0.0
#         # Check intermediate assistant blocks (those with tool calls)
#         for i, assistant_block in enumerate(assistant_blocks[:-1]):
#             if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
#                 think_match = re.search(r'^<think>(.*?)</think>(\s*)<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
#                 if think_match:
#                     format_reward += 0.2
#         # Check the last assistant block
#         last_assistant_block = assistant_blocks[-1]
#         # 方案1: 基础奖励 - 有answer标签就给奖励
#         if has_answer_tags(last_assistant_block):
#             format_reward += 0.3 # 有answer标签的基础奖励
#             # 完整格式奖励 - think + answer
#             think_answer_match = re.search(r'^<think>(.*?)</think>(.*?)<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
#             if think_answer_match:
#                 format_reward += 0.5 # 完整格式的额外奖励
#     except Exception as e:
#         print(f"[DEBUG] Error in compute_score_format: {e}")
#         return 0.0
#     return min(format_reward, 1.0) # 确保不超过1.0

# def compute_score_format(solution_str):
#     """The scoring function for format reward.

#     Args:
#         solution_str: the solution text
    
#     """
#     if solution_str is None:
#         return 0.0
    
#     try:
#         # Perfect format match for the new structure
#         # First <|im_start|>assistant should have <think> and possibly <query>
#         # Then <|im_start|>tool with <knowledge> (can repeat with assistant/tool pairs)
#         # Final <|im_start|>assistant with the answer and <|im_end|>
        
#         # Check for basic structure with <|im_start|>assistant and <|im_end|> tags
#         breakpoint()
#         assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)

#         format_reward = 0.0
        
#         # If no blocks found, return 0
#         if not assistant_blocks:
#             return 0.0
        
#         # Perfect format requires at least one assistant block and matching tool blocks if tool calls exist
#         # Check first assistant block contains <think> tags
#         for i, assistant_block in enumerate(assistant_blocks[:-1]):
#             if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<tool_call>') == 1 and assistant_block.count('</tool_call>') == 1:
#                 think_match = re.search(r'^<think>(.*?)</think>(\s*)<tool_call>(.*?)</tool_call>$', assistant_block, re.DOTALL)
#                 if think_match:

#                     format_reward += 0.5

#         # Check the last assistant block contains <answer> tags
#         if assistant_blocks:  # 确保有至少一个assistant块
#             last_assistant_block = assistant_blocks[-1]
#             think_answer_match = re.search(r'^<think>(.*?)</think>(.*?)<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
#             if think_answer_match:
#                 format_reward += 1.0
#     except Exception as e:
#         print(f"[DEBUG] Error in compute_score_format: {e}")
#         return 0.0
#     return format_reward



def compute_score_format(solution_str):
    """格式奖励函数 - 版本2：更强调answer标签的重要性。
    Args:
        solution_str: the solution text
    """
    if solution_str is None:
        return 0.0
    try:
        # breakpoint()
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        format_reward = 0.0
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        # Check intermediate assistant blocks
        for i, assistant_block in enumerate(assistant_blocks[:-1]):
            if assistant_block.count('<think>') == 1 and assistant_block.count('</think>') == 1 and assistant_block.count('<interaction_prompt>') == 1 and assistant_block.count('</interaction_prompt>') == 1:
                think_match = re.search(r'^<think>(.*?)</think>(\s*)<interaction_prompt>(.*?)</interaction_prompt>$', assistant_block, re.DOTALL)
                if think_match:
                    format_reward += 0.4 # 降低中间块的奖励
        # Check the last assistant block - 重点关注answer
        last_assistant_block = assistant_blocks[-1]
        # 必须有answer标签才能获得主要奖励
        if has_answer_tags(last_assistant_block):
            format_reward += 0.25 # answer标签的主要奖励
            # 检查answer内容是否非空
            answer_content = extract_solution(last_assistant_block)
            if answer_content and len(answer_content.strip()) > 0:
                format_reward += 0.25 # 非空answer内容奖励
            # 完整格式的额外奖励
            think_answer_match = re.search(r'^<think>(.*?)</think>(.*?)<answer>(.*?)</answer>$', last_assistant_block, re.DOTALL)
            if think_answer_match:
                format_reward += 0.1 # 完整格式奖励
        # else:
        #     # 没有answer标签，给予惩罚
        #     format_reward -= 0.5
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_v2: {e}")
        return 0.0
    return format_reward 

def compute_score_answer(solution_str, ground_truth):
    """改进的答案奖励函数 - 只有在有answer标签时才计算答案奖励
    现在支持单个答案和多个答案的list格式
    """
    if solution_str is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        solution_str = assistant_blocks[-1]
        # 只有当存在answer标签时才计算答案奖励
        if not has_answer_tags(solution_str):
            return 0.0
        answer = extract_solution(solution_str)
        answer_reward = 0.0
        if answer is not None:
            # 使用修改后的cal_f1_score函数，支持多答案
            answer_reward = cal_f1_score(answer, ground_truth)
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_answer: {e}")
        return 0.0
    return answer_reward

# def compute_score_format_answer_v3(solution_str, ground_truth):
#     """综合奖励函数 - 版本3：强调answer标签的重要性
#     Args:
#         solution_str: the solution text
#         ground_truth: the ground truth
#     """
#     if solution_str is None or ground_truth is None:
#         return -1.0
#     try:
#         # 检查是否有answer标签
#         has_answer = has_answer_tags(solution_str)
#         if not has_answer:
#             # 没有answer标签，给予重惩罚
#             return -1.0
#         # 有answer标签，计算格式和答案奖励
#         format_reward = compute_score_format_v2(solution_str)
#         answer_reward = compute_score_answer(solution_str, ground_truth)
#         # 综合奖励计算
#         if format_reward >= 0.8: # 格式基本正确
#             rewards = -1.0 + format_reward + answer_reward
#         else:
#             rewards = -1.0 + format_reward * 0.5 # 格式不完整时降低奖励
#         return rewards
#     except Exception as e:
#         print(f"[DEBUG] Error in compute_score_format_answer_v3: {e}")
#         return -1.0

# 保持原有的其他函数不变
def compute_score_sm(solution_str, ground_truth):
    """The scoring function for substring match.
    现在支持单个答案和多个答案的list格式
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        if not assistant_blocks or len(assistant_blocks) == 0:
            return 0.0
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        sm_score = float(subem_check(answer, ground_truth))
        return sm_score
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_sm: {e}")
        return 0.0

def compute_score_em(solution_str, ground_truth):
    """The scoring function for exact match (EM).
    现在支持单个答案和多个答案的list格式
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        # 使用修改后的exact_match_score函数，支持多答案
        em_score = exact_match_score(answer, ground_truth)
        return em_score
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_em: {e}")
        return 0.0

def compute_score_f1(solution_str, ground_truth):
    """The scoring function for F1 score.
    现在支持单个答案和多个答案的list格式
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        assistant_blocks = re.findall(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', solution_str, re.DOTALL)
        solution_str = assistant_blocks[-1]
        answer = extract_solution(solution_str)
        if answer is None:
            return 0.0
        # 使用修改后的cal_f1_score函数，支持多答案
        f1_score = cal_f1_score(answer, ground_truth)
        return f1_score
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_f1: {e}")
        return 0.0

# 原有的综合函数保持不变
def compute_score_format_answer(solution_str, ground_truth):
    """原有的综合奖励函数
    现在支持单个答案和多个答案的list格式
    """
    if solution_str is None or ground_truth is None:
        return 0.0
    try:
        format_reward = compute_score_format(solution_str)
        answer_reward = compute_score_answer(solution_str, ground_truth)
        sm_score_reward = compute_score_sm(solution_str, ground_truth)
        format_reward = min(format_reward, 1.0)
        if format_reward == 1.0:
            rewards = -1.0 + format_reward + answer_reward
            return rewards
        else:
            rewards = -1.0 + format_reward
            return rewards
    except Exception as e:
        print(f"[DEBUG] Error in compute_score_format_answer: {e}")
        return -1.0