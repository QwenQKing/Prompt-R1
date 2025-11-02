# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 在 Apache 2.0 许可下发布；下面是许可证的标准声明，略。
# ...

from collections import defaultdict   # 用于创建默认值为 list 的字典，方便收集额外信息

import torch                          # PyTorch，用于张量计算

from verl import DataProto            # veRL 的数据容器：包含 .batch(张量) 和 .non_tensor_batch(非张量)
from agent_r1.src.reward_score import _default_compute_score  # 默认的打分函数（格式/答案等组合）

class AgentRewardManager:
    """The reward manager.
    奖励管理器：负责从模型生成的序列中解码、组装成字符串，调用打分函数，返回逐 token 的奖励张量。
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer               # 用于把 token id 列表解码为字符串
        self.num_examine = num_examine           # 每个数据来源(data_source)最多打印多少条样例（便于调试）
        self.compute_score = compute_score or _default_compute_score  # 外部注入或使用默认打分函数
        self.reward_fn_key = reward_fn_key       # 从 non_tensor_batch 中取“数据来源”的键名，默认 "data_source"

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets
        调用实例本身即执行“计算奖励”的流程。
        参数:
          - data: DataProto，包含当前 batch 的张量与非张量信息
          - return_dict: 若为 True，返回包含奖励张量与额外信息的 dict；否则只返回奖励张量
        """

        # 如果 batch 里已经有“现成的 rm_scores”（例如外部评分模型预先算好的分数），直接返回它
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        # 初始化奖励张量，全 0，形状与 responses 相同（逐 token 位置的奖励容器）
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        # 用于汇总额外信息（例如 acc、format 分等），默认每个 key 是一个 list
        reward_extra_info = defaultdict(list)

        # 记录各个 data_source 已经打印了多少条，防止控制台刷屏
        already_print_data_sources = {}

        # 遍历 batch 中的每个样本（DataProto 支持下标访问）
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem：单样本视图，含 .batch 与 .non_tensor_batch

            # 取出该样本的 prompts（注意：是最终输出里保存的“只含 prompt 段”的 token ids）
            prompt_ids = data_item.batch["prompts"]

            # prompt 的长度（序列维度长度）
            prompt_length = prompt_ids.shape[-1]

            # 计算有效的 prompt 长度：用 attention_mask 的前 prompt_length 段求和（去掉 padding）
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            # 只保留右侧有效部分（如果 prompt 前面有左侧 padding，这里用负索引截取）
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # 取出 responses（包含模型与工具拼接后的响应 token 段）
            response_ids = data_item.batch["responses"]
            # 计算有效的 response 长度：attention_mask 在 prompt 之后的那部分求和
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            # 只保留 response 的有效前缀（去掉末尾 padding）
            valid_response_ids = response_ids[:valid_response_length]

            # 将有效的 prompt 与 response 拼接成完整序列（用于解码成字符串打分）
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))

            # 解码为字符串（保留特殊 token，便于依赖 <|im_start|>/</think>/<answer> 的正则评分）
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
            pad_token_id = self.tokenizer.pad_token_id
            # 安全处理：按 pad_token 切一次，去掉其后的部分（避免残留 padding 影响评分）
            sequences_str = sequences_str.split(self.tokenizer.decode([pad_token_id]))[0]
            # 末尾若没有 eos_token，则补上（保证评分逻辑的边界一致性）
            if not sequences_str.endswith(self.tokenizer.eos_token):
                sequences_str += self.tokenizer.eos_token

            # 从 non_tensor_batch 里取 ground_truth（参考答案/真值），供评分函数使用
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # 从 non_tensor_batch 里取 data_source（决定采用哪类评分策略/加权等）
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            # 可选的额外信息（可能包含一些数据标签、难度、元数据等）
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            # 调用评分函数，得到分数或包含多项指标的字典
            # 约定：compute_score 返回 float 或 dict（如 {'score': -0.5, 'acc': 0.0, 'format': 0.5}）
            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            # 如果评分函数返回 dict，则主奖励取 "score"，并把其它键值累计到 reward_extra_info
            if isinstance(score, dict):
                reward = score["score"]
                # 存储所有子指标，便于统计/日志
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                # 否则 score 就是一个纯浮点奖励
                reward = score

            # 只在“最后一个有效 response token 位置”写入奖励
            # 这样做的常见原因：把整条序列的奖励（句级）赋在序列末 token 上，便于 RL 训练的 credit assignment
            reward_tensor[i, valid_response_length - 1] = reward

            # 初始化该 data_source 的计数器（若未出现过则置 0）
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # 按 data_source 限制打印次数（调试用）：只打印前 num_examine 条
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                # 打印“prompt+response”整体字符串（包含特殊标记）
                print("[prompt+response]", sequences_str)
                # 打印标准答案
                print("[ground_truth]", ground_truth)
                # 如果是 dict，则逐项打印指标；否则打印单一分数
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # 根据入参决定返回内容：仅奖励张量，或包含额外信息的 dict
        if return_dict:
            return {
                "reward_tensor": reward_tensor,        # 逐 token 奖励（除末 token，一般为 0）
                "reward_extra_info": reward_extra_info # 收集的各类指标（列表形式）
            }
        else:
            return reward_tensor
