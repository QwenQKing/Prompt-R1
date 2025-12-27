#!/usr/bin/env python3
"""
批量推理脚本 - 使用batch_step进行真正的批量处理（修复JSON序列化问题）
"""

import json
import os
import re
import copy
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from openai import OpenAI
from datetime import datetime

from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config

import debugpy

# # 启动调试服务器并指定端口
# debugpy.listen(('0.0.0.0', 8006))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # 等待调试器连接


def convert_text_to_json_format(response_str):
    """将纯文本格式的 <interaction_prompt> 转换成 JSON 格式"""
    pattern = r'<interaction_prompt>(.*?)</interaction_prompt>'
    matches = re.findall(pattern, response_str, re.DOTALL)
    
    if not matches:
        return response_str
    
    for match in matches:
        content = match.strip()
        
        try:
            json.loads(content)
            continue
        except json.JSONDecodeError:
            json_content = json.dumps({
                "name": "prompt",
                "arguments": {
                    "prompt": content
                }
            })
            
            old_tag = f"<interaction_prompt>{match}</interaction_prompt>"
            new_tag = f"<interaction_prompt>\n{json_content}\n</interaction_prompt>"
            response_str = response_str.replace(old_tag, new_tag)
    
    return response_str


def process_tool_call(response_str):
    """处理工具调用"""
    eos_token = "<|im_end|>"
    tool_call_end = "</interaction_prompt>"
    
    response_str = convert_text_to_json_format(response_str)
    
    if "<interaction_prompt>" not in response_str:
        return response_str + eos_token, False
    
    if tool_call_end in response_str:
        response_str = response_str.split(tool_call_end)[0] + tool_call_end
    
    return response_str + eos_token, True


def extract_interaction_prompt_content(raw_response):
    """提取 <interaction_prompt> 标签之间的内容"""
    tool_call_start = "<interaction_prompt>"
    tool_call_end = "</interaction_prompt>"
    pattern = re.compile(f"{re.escape(tool_call_start)}(.*?){re.escape(tool_call_end)}", re.DOTALL)
    
    matches = re.findall(pattern, raw_response)
    if matches:
        return matches[0].strip()
    
    return ""


def extract_tool_response_content(tool_response):
    """从工具响应中提取results内容"""
    if not tool_response or tool_response == "":
        return ""
    
    tool_response_start = "<interaction_response>"
    tool_response_end = "</interaction_response>"
    pattern = re.compile(f"{re.escape(tool_response_start)}(.*?){re.escape(tool_response_end)}", re.DOTALL)
    
    matches = re.findall(pattern, tool_response)
    if matches:
        json_content = matches[0].strip()
        try:
            tool_response_data = json.loads(json_content)
            if isinstance(tool_response_data, dict) and "results" in tool_response_data:
                results = tool_response_data["results"]
                if isinstance(results, list) and len(results) > 0:
                    return results[0]
        except json.JSONDecodeError:
            pass
    
    try:
        tool_response_data = json.loads(tool_response)
        if isinstance(tool_response_data, dict) and "results" in tool_response_data:
            results = tool_response_data["results"]
            if isinstance(results, list) and len(results) > 0:
                return results[0]
    except json.JSONDecodeError:
        pass
    
    return ""


def extract_final_answer(response_str):
    """从响应中提取最终答案"""
    answer_match = re.findall(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
    if answer_match:
        return answer_match[0].strip()
    return ""


def format_interaction_log(turn_idx, response_str, tool_response, has_tool_call):
    """格式化交互日志"""
    log_lines = [f"\n{'='*60}", f"Turn {turn_idx + 1}"]
    
    if has_tool_call:
        think_match = re.findall(r'<think>(.*?)</think>', response_str, re.DOTALL)
        prompt_match = re.findall(r'<interaction_prompt>(.*?)</interaction_prompt>', response_str, re.DOTALL)
        
        if think_match:
            log_lines.append(f"\n<Think>\n{think_match[0].strip()}")
        
        if prompt_match:
            prompt_content = prompt_match[0].strip()
            try:
                prompt_json = json.loads(prompt_content)
                if "arguments" in prompt_json and "prompt" in prompt_json["arguments"]:
                    prompt_content = prompt_json["arguments"]["prompt"]
            except:
                pass
            log_lines.append(f"\n<Interaction Prompt>\n{prompt_content}")
        
        if tool_response:
            knowledge_match = re.findall(r'<interaction_response>(.*?)</interaction_response>', tool_response, re.DOTALL)
            if knowledge_match:
                try:
                    knowledge_data = json.loads(knowledge_match[0])
                    knowledge_list = knowledge_data.get('results', [])
                    knowledge = "\n".join(str(k) for k in knowledge_list)
                    log_lines.append(f"\n<Interaction Response>\n{knowledge}")
                except:
                    log_lines.append(f"\n<Interaction Response>\n{knowledge_match[0]}")
    else:
        think_match = re.findall(r'<think>(.*?)</think>', response_str, re.DOTALL)
        answer_match = re.findall(r'<answer>(.*?)</answer>', response_str, re.DOTALL)
        
        if think_match:
            log_lines.append(f"\n<Think>\n{think_match[0].strip()}")
        if answer_match:
            log_lines.append(f"\n<Answer>\n{answer_match[0].strip()}")
    
    return "\n".join(log_lines)


def convert_to_json_serializable(obj):
    """
    将numpy类型转换为Python原生类型，确保JSON可序列化
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class BatchInferenceState:
    """管理批量推理的状态"""
    def __init__(self, batch_size, system_instruction, instruction):
        self.batch_size = batch_size
        self.active_mask = [True] * batch_size  # 哪些样本还在继续
        self.messages = []  # 每个样本的当前messages
        self.conversation_histories = []  # 每个样本的会话历史（标准消息格式）
        self.dialogue_logs = []  # 每个样本的对话日志
        self.final_answers = [None] * batch_size  # 每个样本的最终答案
        
        # 初始化每个样本的状态
        for i in range(batch_size):
            self.messages.append([])
            self.conversation_histories.append([])
            self.dialogue_logs.append([])
    
    def initialize_question(self, idx, question, system_instruction, instruction):
        """初始化单个问题"""
        self.messages[idx] = [{
            "role": "user",
            "content": system_instruction + question + '\n' + instruction
        }]
    
    def update_after_tool_call(self, active_indices, raw_responses, tool_responses):
        """批量更新工具调用后的状态"""
        for i, active_idx in enumerate(active_indices):
            if not self.active_mask[active_idx]:
                continue
            
            response_str = raw_responses[i]
            tool_response = tool_responses[i]
            
            # 提取用户prompt和助手回复
            user_prompt = extract_interaction_prompt_content(response_str)
            if user_prompt:
                try:
                    prompt_data = json.loads(user_prompt)
                    if isinstance(prompt_data, dict) and "arguments" in prompt_data:
                        actual_prompt = prompt_data["arguments"].get("prompt", user_prompt)
                    else:
                        actual_prompt = user_prompt
                except json.JSONDecodeError:
                    actual_prompt = user_prompt
                
                assistant_content = extract_tool_response_content(tool_response)
                
                # 更新会话历史
                self.conversation_histories[active_idx].append({
                    "role": "user",
                    "content": actual_prompt
                })
                self.conversation_histories[active_idx].append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            # 更新messages用于下一轮
            next_step_prompt = self.messages[active_idx][0]["content"] + response_str + tool_response
            self.messages[active_idx] = [{
                "role": "user",
                "content": next_step_prompt
            }]
    
    def get_active_data(self):
        """获取所有活跃样本的数据"""
        active_indices = [i for i, active in enumerate(self.active_mask) if active]
        active_messages = [self.messages[i] for i in active_indices]
        active_histories = [self.conversation_histories[i] for i in active_indices]
        return active_indices, active_messages, active_histories


def estimate_tokens(text: str) -> int:
    """粗略估计文本的token数量（1 token ≈ 4 characters）"""
    return len(text) // 4


def batch_generate(client, messages_list, model_name, temperature, top_p, max_tokens, max_context_length=40960):
    """批量调用OpenAI API生成"""
    responses = []
    for messages in messages_list:
        # 估算当前消息的token数
        total_text = ""
        for msg in messages:
            total_text += msg.get("content", "")
        
        estimated_tokens = estimate_tokens(total_text)
        
        # 如果估算的tokens + max_tokens超过限制，返回错误信息并停止该样本
        if estimated_tokens + max_tokens > max_context_length:
            print(f"[WARNING] 上下文过长 (~{estimated_tokens} tokens)，已达到最大轮数限制")
            responses.append("<answer>Maximum context length exceeded</answer>")
            continue
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            if response and response.choices and response.choices[0].message.content:
                responses.append(response.choices[0].message.content)
            else:
                responses.append("")
        except Exception as e:
            if "maximum context length" in str(e).lower():
                print(f"[ERROR] API返回上下文长度错误: {e}")
                responses.append("<answer>Maximum context length exceeded</answer>")
            else:
                raise e
    
    return responses


def process_batch_questions(
    client, env, questions, system_instruction, instruction,
    max_turns, max_tokens, temperature, top_p, model_name
) -> Tuple[List[str], List[List[str]]]:
    """
    批量处理多个问题，使用batch_step
    
    Returns:
        final_answers: 每个问题的最终答案
        dialogue_logs: 每个问题的对话日志列表
    """
    batch_size = len(questions)
    state = BatchInferenceState(batch_size, system_instruction, instruction)
    
    # 初始化所有问题
    for i, question in enumerate(questions):
        state.initialize_question(i, question, system_instruction, instruction)
    
    # 主循环
    for turn_idx in range(max_turns):
        # 获取活跃样本
        active_indices, active_messages, active_histories = state.get_active_data()
        
        if not active_indices:
            break  # 所有样本都已完成
        
        print(f"  Turn {turn_idx + 1}: {len(active_indices)} 活跃样本")
        
        # 批量生成（带上下文长度检查）
        raw_responses = batch_generate(
            client, active_messages, model_name, 
            temperature, top_p, max_tokens
        )
        
        # 处理每个响应
        processed_responses = []
        has_tool_calls = []
        
        for response_str in raw_responses:
            # 检查是否因为上下文过长而返回特殊答案
            if "Maximum context length exceeded" in response_str:
                # 强制停止该样本
                processed_str = response_str
                has_tool_call = False
            else:
                processed_str, has_tool_call = process_tool_call(response_str)
            
            processed_responses.append(processed_str)
            has_tool_calls.append(has_tool_call)
        
        # 使用batch_step批量处理工具调用
        tool_responses, tool_successes, new_active_masks = env.batch_step(
            processed_responses, 
            active_histories
        )
        
        # 更新状态
        for i, active_idx in enumerate(active_indices):
            # 记录日志
            log_entry = format_interaction_log(
                turn_idx, 
                processed_responses[i], 
                tool_responses[i],
                has_tool_calls[i]
            )
            state.dialogue_logs[active_idx].append(log_entry)
            
            # 更新活跃状态
            if not new_active_masks[i] or "Maximum context length exceeded" in processed_responses[i]:
                state.active_mask[active_idx] = False
                # 提取答案
                answer = extract_final_answer(processed_responses[i])
                if not answer and "Maximum context length exceeded" in processed_responses[i]:
                    answer = "Maximum context length exceeded"
                state.final_answers[active_idx] = answer
        
        # 批量更新工具调用后的状态
        still_active_indices = [active_indices[i] for i, active in enumerate(new_active_masks) 
                               if active and "Maximum context length exceeded" not in processed_responses[i]]
        still_active_responses = [processed_responses[i] for i, active in enumerate(new_active_masks) 
                                 if active and "Maximum context length exceeded" not in processed_responses[i]]
        still_active_tool_responses = [tool_responses[i] for i, active in enumerate(new_active_masks) 
                                      if active and "Maximum context length exceeded" not in processed_responses[i]]
        
        if still_active_indices:
            state.update_after_tool_call(
                still_active_indices,
                still_active_responses,
                still_active_tool_responses
            )
    
    # 为仍然活跃的样本设置默认答案
    for i, active in enumerate(state.active_mask):
        if active and state.final_answers[i] is None:
            state.final_answers[i] = "Maximum turns reached without final answer"
    
    return state.final_answers, state.dialogue_logs


def process_parquet_file(parquet_path, output_dir, client, env, config_params, batch_size=4):
    """处理单个parquet文件，使用批量处理"""
    # 读取parquet文件
    df = pd.read_parquet(parquet_path)
    
    # 创建输出目录
    file_name = Path(parquet_path).stem
    file_output_dir = output_dir / file_name
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 打开日志文件
    log_path = file_output_dir / "log.txt"
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"测试文件: {parquet_path}\n")
        log_file.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"数据总数: {len(df)}\n")
        log_file.write(f"批处理大小: {batch_size}\n")
        log_file.write("="*80 + "\n\n")
        
        # 分批处理
        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]
            
            print(f"\n处理 {file_name} - Batch {batch_start//batch_size + 1} (样本 {batch_start}-{batch_end-1})")
            
            # 准备批次数据
            batch_questions = []
            batch_ground_truths = []
            batch_indices = []
            
            for idx, row in batch_df.iterrows():
                # 提取问题
                prompt_content = row['prompt'][0]['content'] 
                question = prompt_content
                
                # 将ground_truth转换为Python原生列表，避免JSON序列化错误
                ground_truth = row['reward_model']['ground_truth']
                ground_truth = convert_to_json_serializable(ground_truth)
                
                # 确保ground_truth是列表
                if not isinstance(ground_truth, list):
                    ground_truth = [ground_truth] if ground_truth is not None else []
                
                batch_questions.append(question)
                batch_ground_truths.append(ground_truth)
                batch_indices.append(int(idx))
            
            # 批量处理这批问题
            predicted_answers, dialogue_logs = process_batch_questions(
                client=client,
                env=env,
                questions=batch_questions,
                system_instruction=config_params['system_instruction'],
                instruction=config_params['instruction'],
                max_turns=config_params['max_turns'],
                max_tokens=config_params['max_tokens'],
                temperature=config_params['temperature'],
                top_p=config_params['top_p'],
                model_name=config_params['model_name']
            )
            
            # 保存结果和日志
            for i in range(len(batch_questions)):
                result = {
                    "data_id": batch_indices[i],
                    "question": batch_questions[i],
                    "predicted_answer": predicted_answers[i] if predicted_answers[i] else "",
                    "ground_truth": batch_ground_truths[i]
                }
                all_results.append(result)
                
                # 写入日志
                log_file.write(f"\n{'#'*80}\n")
                log_file.write(f"数据ID: {batch_indices[i]}\n")
                log_file.write(f"问题: {batch_questions[i]}\n")
                log_file.write(f"真实答案: {batch_ground_truths[i]}\n")
                log_file.write(f"{'#'*80}\n")
                log_file.write("\n".join(dialogue_logs[i]))
                log_file.write("\n\n" + "="*80 + "\n")
                log_file.write(f"预测答案: {predicted_answers[i]}\n")
                log_file.write("="*80 + "\n\n")
                log_file.flush()
    
    # 保存结果到JSON
    json_path = file_output_dir / "res.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"完成文件 {file_name}，结果保存到 {file_output_dir}")


def main():
    # 配置参数
    TOOLS = default_config.TOOLS if isinstance(default_config.TOOLS, list) else [default_config.TOOLS]
    OPENAI_API_KEY = default_config.OPENAI_API_KEY
    OPENAI_API_BASE = default_config.OPENAI_API_BASE
    MODEL_NAME = default_config.MODEL_NAME
    TEMPERATURE = default_config.TEMPERATURE
    TOP_P = default_config.TOP_P
    MAX_TOKENS = 8192  # 降低单次生成的token数，避免超过上下文限制
    MAX_TURNS = 10  # 减少最大轮数，避免对话过长
    INSTRUCTION = default_config.INSTRUCTION
    system_instruction = default_config.system_instruction
    BATCH_SIZE = 32  # 批处理大小
    
    print(f"[INFO] 配置: MAX_TOKENS={MAX_TOKENS}, MAX_TURNS={MAX_TURNS}, BATCH_SIZE={BATCH_SIZE}")
    
    # 初始化客户端和环境
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    tools = [_default_tool(tool) for tool in TOOLS]
    env = NousToolEnv(tools=tools, max_tool_response_length=MAX_TOKENS)
    
    # 配置参数字典
    config_params = {
        'system_instruction': system_instruction,
        'instruction': INSTRUCTION,
        'max_turns': MAX_TURNS,
        'max_tokens': MAX_TOKENS,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'model_name': MODEL_NAME
    }
    
    # 输入输出路径
    input_dir = Path("eval-2-test")
    output_dir = Path("prompt-r1-1102")
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有parquet文件
    parquet_files = list(input_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"在 {input_dir} 目录下未找到parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    # 处理每个parquet文件
    for parquet_path in parquet_files:
        print(f"\n开始处理文件: {parquet_path}")
        try:
            process_parquet_file(parquet_path, output_dir, client, env, config_params, batch_size=BATCH_SIZE)
        except Exception as e:
            print(f"[ERROR] 处理文件 {parquet_path} 时出错: {e}")
            continue
    
    print("\n所有文件处理完成！")


if __name__ == "__main__":
    main()