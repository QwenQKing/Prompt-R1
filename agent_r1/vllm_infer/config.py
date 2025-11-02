"""
Configuration parameters for the VLLM inference
"""

# Environment and API settings
TOOLS = ["prompt"]
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8082/v1"
MODEL_NAME = "agent"

# Model inference parameters
TEMPERATURE = 0.7
TOP_P = 0.8
MAX_TOKENS = 8192
REPETITION_PENALTY = 1.05

system_instruction = (
        """
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a great assistant. 

# Tools

You can call one or more powerful Large Language Models to answer the user's questions. But you MUST provide the tool with an explanation and analysis of the problem, as well as your thought process.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "prompt", "description": "Give an explanation of the question in detail.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Give question and its explanation"}}, "required": ["prompt"]}}}
</tools>

For each function call, return a json object with function name and arguments within <interaction_prompt></interaction_prompt> XML tags:
<interaction_prompt>
{"name": <function-name>, "arguments": <args-json-object>}
</interaction_prompt><|im_end|>
<|im_start|>\nuser
"""

)

INSTRUCTION = (""
        # "\"You are Qwen, created by Alibaba Cloud. You are a great assistant. \n\nYou can call one or more powerful Large Language Models to answer the user's questions. But you MUST provide the tool with an explanation and analysis of the problem, as well as your thought process.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"prompt\", \"description\": \"Give an explanation of the question in detail.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"prompt\": {\"type\": \"string\", \"description\": \"Give question and its explanation\"}}, \"required\": [\"prompt\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <interaction_prompt></interaction_prompt> XML tags:\n<interaction_prompt>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</interaction_prompt>\""
# """
# First, provide a simple explanation of the question and give it to the large language model for a more accurate answer. Focus on explaining the question without deep reasoning in the first step. After receiving the response, think about the large language model's response, and by interacting with the large language model again and again, arrive at the final answer. Proceed step by step with the following rules:
# 1. Only in the first step, provide a brief explanation of the question and give it to the large language model:
#    <think>(don't think deeply and no more than 50 words)</think>
#    <interaction_prompt>(give the question and its explanation to the large language model)</interaction_prompt>
# 2. After the first step, in each interaction with the large language model, write:
#    <think>(your reasoning for the receiving response and question)</think>
#    <interaction_prompt>(new request to refine or validate the answer)</interaction_prompt>
# 3. Each <interaction_prompt> must build on what came before. Do not just repeat the same content. Let the content of the <interaction_prompt>...</interaction_prompt> evolve naturally (for example: outline → add details → refine → check). 
# 4. Continue producing think within <think></think> and call tool within <interaction_prompt></interaction_prompt> until the answer is ready.
# 5. Once the answer is complete, write:
#    <think>(final reasoning with the <interaction_response> and question)</think>
#    <answer>(final answer for the question)</answer>
# assistant
# """

)
single_system_instruction = (
        """
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a great assistant. 

# Tools

You can call one or more powerful Large Language Models to answer the user's questions. But you MUST provide the tool with an explanation and analysis of the problem, as well as your thought process.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "prompt", "description": "Give an explanation of the question in detail.", "parameters": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Give question and its explanation"}}, "required": ["prompt"]}}}
</tools>

For each function call, return a json object with function name and arguments within <interaction_prompt></interaction_prompt> XML tags:
<interaction_prompt>
{"name": <function-name>, "arguments": <args-json-object>}
</interaction_prompt><|im_end|>
<|im_start|>\nuser
"""

)

single_INSTRUCTION = (""
        "\"You are Qwen, created by Alibaba Cloud. You are a great assistant. \n\nYou can call one or more powerful Large Language Models to answer the user's questions. But you MUST provide the tool with an explanation and analysis of the problem, as well as your thought process.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"prompt\", \"description\": \"Give an explanation of the question in detail.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"prompt\": {\"type\": \"string\", \"description\": \"Give question and its explanation\"}}, \"required\": [\"prompt\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <interaction_prompt></interaction_prompt> XML tags:\n<interaction_prompt>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</interaction_prompt>\""
"""
First, provide a simple explanation of the question and give it to the large language model for a more accurate answer. Focus on explaining the question without deep reasoning in the first step. After receiving the response, think about the large language model's response, and by interacting with the large language model again and again, arrive at the final answer. Proceed step by step with the following rules:
1. Only in the first step, provide a brief explanation of the question and give it to the large language model:
   <think>(don't think deeply and no more than 50 words)</think>
   <interaction_prompt>(give the question and its explanation to the large language model)</interaction_prompt>
2. After the first step, in each interaction with the large language model, write:
   <think>(your reasoning for the receiving response and question)</think>
   <interaction_prompt>(new request to refine or validate the answer)</interaction_prompt>
3. Each <interaction_prompt> must build on what came before. Do not just repeat the same content. Let the content of the <interaction_prompt>...</interaction_prompt> evolve naturally (for example: outline → add details → refine → check). 
4. Continue producing think within <think></think> and call tool within <interaction_prompt></interaction_prompt> until the answer is ready.
5. Once the answer is complete, write:
   <think>(final reasoning with the <interaction_response> and question)</think>
   <answer>(final answer for the question)</answer>
assistant
"""

)