#!/usr/bin/env python3
"""
Script to run inference with configurable parameters
"""

import json
import importlib
import os
import re
import copy
from openai import OpenAI

from agent_r1.tool.envs.nous import NousToolEnv
from agent_r1.tool.tools import _default_tool
import agent_r1.vllm_infer.config as default_config

import debugpy

# # 启动调试服务器并指定端口
# debugpy.listen(('0.0.0.0', 8086))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()  # 等待调试器连接

def load_custom_config(config_path):
    """Load custom configuration from a Python file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    spec = importlib.util.spec_from_file_location("custom_config", config_path)
    custom_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_config)
    return custom_config

def convert_text_to_json_format(response_str):
    """
    将纯文本格式的 <interaction_prompt> 转换成 JSON 格式
    """
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
    """
    Process the response to detect if there's a tool call
    Returns: (processed_response, has_tool_call)
    """
    eos_token = "<|im_end|>"
    tool_call_end = "</interaction_prompt>"
    
    response_str = convert_text_to_json_format(response_str)
    
    if "<interaction_prompt>" not in response_str:
        return response_str + eos_token, False
    
    if tool_call_end in response_str:
        response_str = response_str.split(tool_call_end)[0] + tool_call_end
    
    return response_str + eos_token, True

def extract_interaction_prompt_content(raw_response):
    """
    直接提取 <interaction_prompt> 标签之间的完整内容作为prompt
    这是要传给工具的原始内容，也是要记录到会话历史的用户消息
    """
    tool_call_start = "<interaction_prompt>"
    tool_call_end = "</interaction_prompt>"
    pattern = re.compile(f"{re.escape(tool_call_start)}(.*?){re.escape(tool_call_end)}", re.DOTALL)
    
    matches = re.findall(pattern, raw_response)
    if matches:
        return matches[0].strip()
    
    return ""

def extract_tool_response_content(tool_response):
    """
    从格式化的工具响应中提取results数组的内容
    """
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
    
    # 尝试直接解析JSON
    try:
        tool_response_data = json.loads(tool_response)
        if isinstance(tool_response_data, dict) and "results" in tool_response_data:
            results = tool_response_data["results"]
            if isinstance(results, list) and len(results) > 0:
                return results[0]
    except json.JSONDecodeError:
        pass
    
    return ""

def print_interaction(mode, r_str, t_str):
    """
    Print interaction output
    mode: True if tool call, False if final answer
    """
    if not r_str.startswith("<think>"):
        r_str = "<think>\n" + r_str
    
    if mode:
        # Tool call mode
        think_match = re.findall(r'<think>(.*?)</think>', r_str, re.DOTALL)
        prompt_match = re.findall(r'<interaction_prompt>(.*?)</interaction_prompt>', r_str, re.DOTALL)
        
        think = think_match[0] if think_match else ""
        
        if prompt_match:
            prompt_content = prompt_match[0].strip()
            try:
                prompt_json = json.loads(prompt_content)
                if "arguments" in prompt_json and "prompt" in prompt_json["arguments"]:
                    prompt_content = prompt_json["arguments"]["prompt"]
            except:
                pass
        else:
            prompt_content = ""
        
        knowledge = ""
        if t_str:
            knowledge_match = re.findall(r'<interaction_response>(.*?)</interaction_response>', t_str, re.DOTALL)
            if knowledge_match:
                try:
                    knowledge_data = json.loads(knowledge_match[0])
                    knowledge_list = knowledge_data.get('results', [])
                    knowledge = "\n".join(str(k) for k in knowledge_list)
                except:
                    knowledge = knowledge_match[0]
        
        print(f"\n[Think]\n{think}")
        print(f"\n[Interaction Prompt]\n{prompt_content}")
        if knowledge:
            print(f"\n[Interaction Response]\n{knowledge}")
    else:
        # Answer mode
        think_match = re.findall(r'<think>(.*?)</think>', r_str, re.DOTALL)
        answer_match = re.findall(r'<answer>(.*?)</answer>', r_str, re.DOTALL)
        
        think = think_match[0] if think_match else ""
        answer = answer_match[0] if answer_match else ""
        
        print(f"\n[Think]\n{think}")
        print(f"\n[Answer]\n{answer}\n")

def main():
    # Default values from config
    TOOLS = default_config.TOOLS if isinstance(default_config.TOOLS, list) else [default_config.TOOLS]
    OPENAI_API_KEY = default_config.OPENAI_API_KEY
    OPENAI_API_BASE = default_config.OPENAI_API_BASE
    MODEL_NAME = default_config.MODEL_NAME
    TEMPERATURE = default_config.TEMPERATURE
    TOP_P = default_config.TOP_P
    MAX_TOKENS = 8192
    REPETITION_PENALTY = default_config.REPETITION_PENALTY
    MAX_TURNS = 5
    QUESTION = "What is Gemini?"
    INSTRUCTION = default_config.single_INSTRUCTION
    system_instruction = default_config.single_system_instruction
    
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
    )
    
    tools = []
    for tool in TOOLS:
        tools.append(_default_tool(tool))
    env = NousToolEnv(tools=tools, max_tool_response_length=MAX_TOKENS)
    
    question_raw = QUESTION
    messages = [{
        "role": "user",
        "content":  system_instruction + "Question: " + question_raw + '\n' + INSTRUCTION
    }]
    
    conversation_history = []
    
    for step in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
        )
        
        if not response or not response.choices or not response.choices[0].message.content:
            break
        
        response_str = response.choices[0].message.content
        response_str, has_tool_call = process_tool_call(response_str)
        tool_response, tool_successes, active = env.step(response_str, conversation_history)
        print_interaction(has_tool_call, copy.deepcopy(response_str), copy.deepcopy(tool_response))
        
        if has_tool_call:
            assistant_content = extract_tool_response_content(tool_response)
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
                
                conversation_history.append({
                    "role": "user",
                    "content": actual_prompt
                })
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_content
                })
            
            next_step_prompt = messages[0]["content"] + response_str + tool_response
            messages = [{
                "role": "user",
                "content": next_step_prompt
            }]
        else:
            break
    
if __name__ == "__main__":
    main()