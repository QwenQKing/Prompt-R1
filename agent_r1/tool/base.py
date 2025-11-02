from abc import ABC, abstractmethod                 # ABC/abstractmethod：用于定义抽象基类和抽象方法
from typing import Dict, List, Any, Tuple           # 类型注解
from agent_r1.tool.utils import is_tool_schema      # 自定义校验函数：检查工具描述是否符合 OpenAI 风格 JSON Schema
from PIL import Image                               # 用于图像类型（图像工具环境）
from jsonschema import validate, ValidationError    # 运行时校验参数是否符合 JSON Schema
import torch                                        # 用于 batch 处理时对 token ids 的潜在处理

class BaseTool(ABC):                                # ——“工具”的抽象基类：具体工具都要继承它
    name: str = ''                                  # 工具名（函数调用时的 name）
    description: str = ''                           # 工具说明（告诉模型什么时候用）
    parameters: dict = {}                           # 工具参数的 JSON Schema（OpenAI 兼容）

    def __init__(self):
        # 初始化时做一些基本合法性校验
        if not self.name:
            raise ValueError('Tool name must be provided')  # 工具名必填

        # 检查 {name, description, parameters} 是否符合“工具模式（schema）”
        # 这样可以在启动时及早发现 schema 写错的问题
        if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
            raise ValueError(
                'The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def execute(self, args: Dict, **kwargs) -> Dict[str, Any]:
        # 抽象方法：执行一次工具调用
        # 子类必须实现。返回值通常约定为 {"content": ..., "success": bool, ...}
        pass
    
    def batch_execute(self, args_list: List[Dict], **kwargs) -> List[Dict[str, Any]]:
        # 批量执行：默认就是一条条调用 execute
        # 子类可重写为更高效的向量化/并发实现
        return [self.execute(args, **kwargs) for args in args_list]
    
    @property
    def tool_info(self) -> Dict:
        # 返回工具的“静态描述”，常用于注册/展示
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }
    
    @property
    def tool_description(self) -> Dict:
        # 返回与 OpenAI “function calling” 兼容的描述结构
        # 上层把这个塞进系统/工具描述，让模型知道如何调用该工具
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
    
    def validate_args(self, args: Dict) -> bool:
        # 运行时对“具体一次调用的参数”进行 JSON Schema 校验
        # 如果不通过，返回 False（或你也可以在外层抛错/打日志）
        try:
            validate(instance=args, schema=self.parameters)
            return True
        except ValidationError:
            return False


class BaseToolEnv(ABC):                             # ——“工具环境”的抽象基类：把 LLM 输出→工具调用→工具返回 串起来
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[bool], bool]:
        """
        工具环境的“状态转移函数”。一次 step 接收 LLM 的原始输出字符串，
        解析其中的工具调用（可能有多个），执行后返回工具响应等信号。

        Args:
            raw_response: LLM 的原始输出文本（可能包含工具调用的 JSON 等）

        Returns:
            tool_response: str
                - 把工具调用的结果组织成“可追加给对话/上下文”的文本（给模型继续读）
            success: List[bool]
                - 每个工具调用是否成功的布尔列表（用于奖励/日志）
            active: bool
                - 轨迹是否继续（False 表示已经终止，比如模型给出最终答案）
        """
        pass

    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[bool]], List[bool]]:
        # 批量 step：对一组 LLM 输出做同样处理
        results = [self.step(raw_response) for raw_response in raw_responses]
        # 组装批量返回（把每个 tuple 对应位置提出来）
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results]
    
    def process_responses_ids(self, tokenizer, raw_responses_ids: torch.Tensor) -> torch.Tensor:
        # 可选：对 LLM 的 token id 序列做二次处理（如剔除特殊标记）
        # 默认不做处理，直接原样返回
        return raw_responses_ids

    @abstractmethod
    def stop(self, raw_response: str) -> bool:
        # 判停函数：给一段 LLM 输出，判断是否达到“终止条件”
        # 例如检测到 "FINAL_ANSWER:" 或者无工具再调用等
        pass

    @abstractmethod
    def extract_tool_calls(self, raw_response: str) -> List[Any]:
        # 从 LLM 输出中抽取“工具调用”列表
        # 具体格式由你定义（JSON 片段 / 特殊标签 / 函数调用对象等）
        pass
    
    @abstractmethod
    def format_tool_response(self, tool_response: str) -> str:
        # 把多个工具的原始返回合并/格式化，转成能再次喂回 LLM 的文本
        # 例如加上 "OBSERVATION:\n..." 之类的标记
        pass

    @property
    def system_prompt(self) -> str:
        # 工具环境建议的 system prompt（可选）
        # 例如告知模型：如何声明工具调用、返回如何被展示、什么时候停止等
        return ""


class BaseImageToolEnv(BaseToolEnv, ABC):           # ——“图像工具环境”的抽象基类：返回里包含 Image 对象的列表
    @abstractmethod
    def step(self, raw_response: str) -> Tuple[str, List[Image.Image], List[bool], bool]:
        # 覆盖 step 的签名：除了文本响应外，还要返回图像列表
        pass
    
    def batch_step(self, raw_responses: List[str]) -> Tuple[List[str], List[List[Image.Image]], List[List[bool]], List[bool]]:
        # 图像环境的批量 step：注意这里多了一维图像列表
        results = [self.step(raw_response) for raw_response in raw_responses]
        return [result[0] for result in results], [result[1] for result in results], [result[2] for result in results], [result[3] for result in results]

# 这份基类在整体框架里的作用（简述）
# BaseTool：

# 约束“一个工具”的最小接口（execute / batch_execute）和描述（name/description/parameters）。

# 内置 JSON Schema 校验，确保你在“函数调用”场景（OpenAI 风格）时，模型知道该怎么传入参数。

# 统一返回结构，便于 Agent 记录日志、做奖励、再拼回上下文。

# BaseToolEnv：

# 处理“LLM 输出 → 工具调用 → 工具执行 → 观察拼接 → 再喂回 LLM”的中间层。

# step 是一次交互；stop 判终止；extract_tool_calls 负责从模型文本里解析出工具请求；format_tool_response 把工具的 observation 格式化。

# GRPO 训练里，rollout 时通常会循环 generate → env.step → append obs → generate... 直到 stop=True，最后算奖励。

# BaseImageToolEnv：

# 和 BaseToolEnv 类似，但适配“会产生图像观察”的工具链（比如视觉检索/渲染），多返回一个 List[Image.Image]。

# 如果你想，我可以基于你现在的输出协议（比如你们的“工具调用 JSON 格式”和“终止条件”）给出一个最小可用的 ToolEnv 子类，直接能在 GRPO rollout 里跑起来。