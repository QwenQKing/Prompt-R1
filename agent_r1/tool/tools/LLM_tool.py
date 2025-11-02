from typing import Dict, List, Any  # 类型注解支持
import os  # 操作系统路径、文件操作
from agent_r1.tool.base import BaseTool  # 导入Agent-R1定义的工具基类
import faiss  # Facebook AI的相似度搜索库，用于加载和查询向量索引
from FlagEmbedding import FlagAutoModel  # FlagEmbedding库，用于加载文本嵌入模型
import json  # JSON编解码
import asyncio
from openai import AsyncOpenAI  # 改用异步客户端

# API_KEY = "sk-RoNKdBYNaF4OyyLH1d06767012634f0e9eF4323f42F93a3f"  # 替换为你的 OpenAI API Key


# client = openai.OpenAI(api_key = API_KEY, base_url = "https://api.apiyi.com/v1")
# API_KEY = "EMPTY"
# MODEL = "gpt-oss-20b"
# BASE_URL = "http://192.168.80.1:8006/v1"

class LLM_Tool(BaseTool):
    # 工具的基本元数据
    name = "prompt"  # 工具名称（Agent调用时识别用）
    description = "Give an explanation of the question in detail."       # 工具功能描述
    parameters = {
        "type": "object",  # 输入参数类型是对象
        "properties": {    # 对象的属性
            "prompt": {"type": "string", "description": "Give question and its explanation"}  # 必须有一个prompt字段（字符串）
        },
        "required": ["prompt"]  # prompt是必填项
    }
        
    def __init__(self):
        super().__init__()
        # 初始化异步OpenAI客户端
        self.client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
        # self.client = AsyncOpenAI(api_key=API_KEY)
        # print("[DEBUG] ASYNC GPT CLIENT INITIALIZED")

    def execute(self, args: Dict) -> Dict[str, Any]:
        """
        执行一次查询 -> 交给GPT-4o-mini，支持标准消息格式的会话历史
        """
        return asyncio.run(self._execute_async(args))
    
    async def _execute_async(self, args: Dict) -> Dict[str, Any]:
        """
        异步执行单次查询，支持标准消息格式的会话历史
        """
        try:
            prompt = args["prompt"]
            
            # ==== 新增：获取标准消息格式的会话历史（如果有的话） ====
            conversation_messages = args.get("conversation_messages", [])
            
            # print(f"[DEBUG] SearchTool.execute - prompt length: {len(prompt)}")
            # print(f"[DEBUG] SearchTool.execute - has conversation_messages: {bool(conversation_messages)}")
            # if conversation_messages:
            #     print(f"[DEBUG] SearchTool.execute - message count: {len(conversation_messages)}")
            #     print(f"[DEBUG] SearchTool.execute - message types: {[msg.get('role', 'unknown') for msg in conversation_messages]}")
            # ==== 标准消息格式会话历史获取结束 ====

            # ==== 新增：构建包含历史上下文的完整消息列表 ====
            messages = []
            
            # 添加系统消息
            messages.append({
                "role": "system", 
                "content": "You are a helpful assistant. Please read the provided content (including previous conversations and the current task) and help the user complete the task or answer the question. "
            })
            
            # 添加历史消息（如果有的话）
            if conversation_messages:
                # 验证并添加历史消息
                for msg in conversation_messages:
                    if isinstance(msg, dict) and "role" in msg and "content" in msg:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                    else:
                        # print(f"[WARNING] SearchTool: Invalid message format: {msg}")
                        pass
                # print(f"[DEBUG] SearchTool.execute - added {len(conversation_messages)} history messages")
            
            # 添加当前用户问题
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # print(f"[DEBUG] SearchTool.execute - total messages: {len(messages)}")
            # ==== 完整消息列表构建结束 ====

            # 异步调用 GPT-4o-mini，直接使用标准消息格式
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=messages
            )

            # 提取回答文本
            answer = response.choices[0].message.content.strip()
            # print(f"[DEBUG] SearchTool.execute - GPT response length: {len(answer)}")

            # 保持与原先一致的返回格式
            return {"content": json.dumps({"results": [answer]}), "success": True}
        except Exception as e:
            # print(f"[ERROR] SearchTool.execute failed: {str(e)}")
            return {"content": str(e), "success": False}

    def batch_execute(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        批量执行 -> 每个prompt分别交给GPT，支持标准消息格式的会话历史
        """
        # print(f"[DEBUG] SearchTool.batch_execute - processing {len(args_list)} requests")
        return asyncio.run(self._batch_execute_async(args_list))
    
    async def _batch_execute_async(self, args_list: List[Dict]) -> List[Dict[str, Any]]:
        """
        异步批量执行，使用并发提升速度，支持标准消息格式的会话历史
        """
        async def _single_request(args, request_idx):
            try:
                prompt = args["prompt"] #20 ={'prompt': 'The question require...lications.'}
                
                # ==== 新增：获取标准消息格式的会话历史（如果有的话） ====
                conversation_messages = args.get("conversation_messages", [])
                
                # print(f"[DEBUG] Request {request_idx}: prompt length={len(prompt)}, has_messages={bool(conversation_messages)}")
                # if conversation_messages:
                #     print(f"[DEBUG] Request {request_idx}: message count={len(conversation_messages)}")
                #     print(f"[DEBUG] Request {request_idx}: message roles={[msg.get('role', 'unknown') for msg in conversation_messages]}")
                # ==== 标准消息格式会话历史获取结束 ====

                # ==== 新增：构建包含历史上下文的完整消息列表 ====
                messages = []
                
                # 添加系统消息
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful assistant. Please read the provided content (including previous conversations and the current task) and help the user complete the task or answer the question. "
                })
                
                # 添加历史消息（如果有的话）
                if conversation_messages:
                    # 验证并添加历史消息
                    for msg_idx, msg in enumerate(conversation_messages):
                        if isinstance(msg, dict) and "role" in msg and "content" in msg:
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        else:
                            # print(f"[WARNING] Request {request_idx}: Invalid message {msg_idx}: {msg}")
                            pass
                    # print(f"[DEBUG] Request {request_idx}: Added {len(conversation_messages)} history messages")
                
                # 添加当前用户问题
                messages.append({
                    "role": "user",
                    "content": prompt
                })
                
                # print(f"[DEBUG] Request {request_idx}: Total messages={len(messages)}")
                # ==== 完整消息列表构建结束 ====

                response = await self.client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                answer = response.choices[0].message.content.strip()
                # print(f"[DEBUG] Request {request_idx}: GPT response length={len(answer)}")
                return {"content": json.dumps({"results": [answer]}), "success": True}
            except Exception as e:
                # print(f"[ERROR] Request {request_idx} failed: {str(e)}")
                return {"content": str(e), "success": False}
        
        # ==== 新增：打印批量执行的调试信息 ====
        # message_count = sum(len(args.get("conversation_messages", [])) for args in args_list)
        # requests_with_messages = sum(1 for args in args_list if args.get("conversation_messages"))
        # print(f"[DEBUG] SearchTool.batch_execute - {requests_with_messages}/{len(args_list)} requests have conversation messages")
        # print(f"[DEBUG] SearchTool.batch_execute - total historical messages: {message_count}")
        
        # for i, args in enumerate(args_list[:3]):  # 只打印前3个请求的详情
        #     messages = args.get("conversation_messages", [])
        #     has_messages = bool(messages)
        #     message_roles = [msg.get('role', 'unknown') for msg in messages] if messages else []
        #     print(f"[DEBUG] Request {i}: has_messages={has_messages}, roles={message_roles}")
        # ==== 批量执行调试信息结束 ====
        
        # 并发执行所有请求
        tasks = [_single_request(args, i) for i, args in enumerate(args_list)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # print(f"[ERROR] Request {i} had exception: {str(result)}")
                processed_results.append({"content": str(result), "success": False})
            else:
                processed_results.append(result)
        
        # print(f"[DEBUG] SearchTool.batch_execute completed - {len(processed_results)} results")
        return processed_results

# 未知提供商：要求显式设置 LLM_MODEL