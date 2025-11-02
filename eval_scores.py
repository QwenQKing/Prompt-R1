#!/usr/bin/env python3
"""
批量评估指标计算脚本 - 遍历prompt-r1-eval-results文件夹，计算指标并汇总到CSV
"""

import re
import string
import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
FILE_PATH="prompt-r1-eval-results"

# ======= 标准化函数 =======
def normalize_answer(s: str) -> str:
    """标准化答案：去除冠词、标点符号、空格及大小写转换"""
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


# ======= F1分数计算 =======
def f1_score(pred: str, gold: str) -> float:
    """计算F1分数"""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    
    g_counts: Dict[str, int] = {}
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1
        
    common = 0
    for t in p_tokens:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    
    if common == 0:
        return 0.0
    
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    
    return 2 * precision * recall / (precision + recall)


# ======= 精确匹配计算 =======
def exact_match(pred: str, gold: str) -> int:
    """计算精确匹配（0或1）"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    return 1 if pred_norm == gold_norm else 0


# ======= 子字符串匹配计算 =======
def substring_match(pred: str, gold: str) -> int:
    """计算子字符串匹配（0或1）"""
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    
    if (gold_norm in pred_norm) or (pred_norm in gold_norm):
        return 1
    return 0


# ======= 语义相似度计算 =======
class SemanticSimilarity:
    """语义相似度计算器（支持多种后端）"""
    def __init__(self):
        self.backend = "auto"
        try:
            from sentence_transformers import SentenceTransformer, util as st_util
            self.backend = "sbert"
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.st_util = st_util
            print("[INFO] 使用sentence-transformers计算语义相似度")
        except Exception:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                self.backend = "tfidf"
                self.vectorizer = TfidfVectorizer()
                self.cosine_similarity = cosine_similarity
                print("[INFO] 使用TF-IDF计算语义相似度")
            except Exception:
                self.backend = "none"
                print("[INFO] 使用基于词汇重叠的简单相似度")

    def pair(self, a: str, b: str) -> float:
        """计算两个字符串的语义相似度"""
        a = a.strip()
        b = b.strip()

        if not a or not b:
            return 1.0
        
        if self.backend == "sbert":
            ea = self.model.encode([a], convert_to_tensor=True, normalize_embeddings=True)
            eb = self.model.encode([b], convert_to_tensor=True, normalize_embeddings=True)
            sim = self.st_util.cos_sim(ea, eb).item()
            return float(sim)
        elif self.backend == "tfidf":
            try:
                X = self.vectorizer.fit_transform([a, b])
                if X.shape[1] == 0:
                    return 1.0
                sim = self.cosine_similarity(X[0], X[1]).item()
                return float(sim)
            except ValueError:
                return 1.0
        else:
            # 基于词汇重叠的简单相似度
            sa = set(normalize_answer(a).split())
            sb = set(normalize_answer(b).split())
            if not sa or not sb:
                return 1.0
            inter = len(sa & sb)
            union = len(sa | sb)
            return inter / union if union > 0 else 0.0


# ======= 批量评估函数 =======
def evaluate_batch(results: List[Dict[str, Any]], compute_semantic: bool = True) -> Dict[str, Any]:
    """
    批量评估多个预测结果
    
    Args:
        results: 包含predicted_answer和ground_truth的字典列表
        compute_semantic: 是否计算语义相似度（默认True）
        
    Returns:
        总体评估指标
    """
    # 只在需要时初始化语义相似度计算器
    sim_calculator = None
    if compute_semantic:
        sim_calculator = SemanticSimilarity()
    
    em_sum = 0
    sm_sum = 0
    f1_list = []
    sem_list = []
    
    # 添加进度条
    for item in tqdm(results, desc="  评估样本", leave=False):
        pred = item.get("predicted_answer", "")
        golds = item.get("ground_truth", [])
        
        if not isinstance(golds, list):
            golds = [str(golds)]
        
        # 计算该样本的最大指标
        max_em = 0
        max_sm = 0
        max_f1 = 0.0
        max_semantic = 0.0
        
        for gold in golds:
            max_em = max(max_em, exact_match(pred, gold))
            max_sm = max(max_sm, substring_match(pred, gold))
            max_f1 = max(max_f1, f1_score(pred, gold))
            
            # 仅在需要时计算语义相似度
            if compute_semantic and sim_calculator:
                max_semantic = max(max_semantic, sim_calculator.pair(pred, gold))
        
        em_sum += max_em
        sm_sum += max_sm
        f1_list.append(max_f1)
        
        if compute_semantic:
            sem_list.append(max_semantic)
    
    n = len(results)
    
    # 构建结果字典
    metrics = {
        "count": n,
        "exact_match": {
            "mean": em_sum / n if n > 0 else 0.0,
            "sum": em_sum
        },
        "substring_match": {
            "mean": sm_sum / n if n > 0 else 0.0,
            "sum": sm_sum
        },
        "f1": {
            "mean": sum(f1_list) / n if n > 0 else 0.0
        }
    }
    
    # 仅在计算了语义相似度时添加该字段
    if compute_semantic:
        metrics["semantic_similarity"] = {
            "mean": sum(sem_list) / n if n > 0 else 0.0
        }
    
    return metrics


def process_single_dataset(res_json_path: Path, compute_semantic: bool) -> Dict[str, Any]:
    """
    处理单个数据集的res.json文件
    
    Returns:
        评估指标字典
    """
    try:
        with open(res_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取文件失败: {e}")
        return None
    
    if not isinstance(data, list):
        print(f"[ERROR] JSON文件应包含一个列表")
        return None
    
    # 计算评估指标
    metrics = evaluate_batch(data, compute_semantic=compute_semantic)
    
    # 保存eval.json到同一目录
    eval_json_path = res_json_path.parent / "eval.json"
    with open(eval_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="批量计算评估指标并汇总到CSV")
    parser.add_argument("--res-dir", type=str, default=FILE_PATH, help="prompt-r1-eval-results")
    parser.add_argument("--output", type=str, default=None, help="输出CSV文件名（默认：自动生成）")
    parser.add_argument("--no-semantic", action="store_true", help="跳过语义相似度计算")
    parser.add_argument("--model-name", type=str, default=FILE_PATH, help="模型名称（用于CSV中的LLM列，默认：从文件夹结构推断）")
    args = parser.parse_args()
    
    res_dir = Path(args.res_dir)
    compute_semantic = not args.no_semantic
    
    if not res_dir.exists():
        print(f"[ERROR] 目录不存在: {res_dir}")
        return
    
    # 收集所有结果
    all_results = []
    
    # 遍历res-eval下的所有数据集文件夹
    dataset_folders = [d for d in res_dir.iterdir() if d.is_dir()]
    
    if not dataset_folders:
        print(f"[ERROR] 在 {res_dir} 下未找到任何数据集文件夹")
        return
    
    print(f"[INFO] 找到 {len(dataset_folders)} 个数据集文件夹")
    print("="*80)
    
    # 使用tqdm遍历数据集文件夹
    for dataset_folder in tqdm(sorted(dataset_folders), desc="处理数据集"):
        res_json = dataset_folder / "res.json"
        
        if not res_json.exists():
            tqdm.write(f"[WARN] 跳过（未找到res.json）: {dataset_folder.name}")
            continue
        
        # 处理单个数据集
        metrics = process_single_dataset(res_json, compute_semantic)
        
        if metrics is None:
            continue
        
        # 推断模型名称（如果未指定）
        if args.model_name:
            model_name = args.model_name
        else:
            # 尝试从父目录结构推断（如果有的话）
            model_name = "unknown-model"
        
        # 数据集名称
        dataset_name = dataset_folder.name
        
        # 构建CSV行
        row = {
            "LLM": model_name,
            "dataset": dataset_name,
            "count": metrics["count"],
            "exact_match_mean": metrics["exact_match"]["mean"],
            "exact_match_sum": metrics["exact_match"]["sum"],
            "substring_match_mean": metrics["substring_match"]["mean"],
            "substring_match_sum": metrics["substring_match"]["sum"],
            "f1_mean": metrics["f1"]["mean"],
        }
        
        if compute_semantic:
            row["semantic_similarity_mean"] = metrics["semantic_similarity"]["mean"]
        
        all_results.append(row)
        
        # 使用tqdm.write避免破坏进度条
        tqdm.write(f"✓ {dataset_name}: EM={metrics['exact_match']['mean']:.4f}, F1={metrics['f1']['mean']:.4f}")
    
    print("="*80)
    
    if not all_results:
        print("[ERROR] 没有成功处理任何数据集")
        return
    
    # 生成CSV文件名
    if args.output:
        csv_filename = args.output
    else:
        model_name = args.model_name if args.model_name else "model"
        csv_filename = f"{model_name}.csv"
    
    csv_path = res_dir / csv_filename
    
    # 写入CSV
    fieldnames = ["LLM", "dataset", "count", "exact_match_mean", "exact_match_sum", 
                  "substring_match_mean", "substring_match_sum", "f1_mean"]
    if compute_semantic:
        fieldnames.append("semantic_similarity_mean")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    print("="*80)
    print(f"[SUCCESS] 汇总结果已保存到: {csv_path}")
    print(f"[INFO] 共处理 {len(all_results)} 个数据集")
    
    # 打印汇总统计
    print("\n汇总统计:")
    print(f"  平均EM: {sum(r['exact_match_mean'] for r in all_results) / len(all_results):.4f}")
    print(f"  平均F1: {sum(r['f1_mean'] for r in all_results) / len(all_results):.4f}")
    if compute_semantic:
        print(f"  平均语义相似度: {sum(r['semantic_similarity_mean'] for r in all_results) / len(all_results):.4f}")


if __name__ == "__main__":
    main()