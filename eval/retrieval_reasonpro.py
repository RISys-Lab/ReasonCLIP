import pandas as pd
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import json
from typing import List, Dict, Tuple
import argparse
from transformers import AutoProcessor, AutoModel, CLIPModel, CLIPProcessor
import open_clip


class ModelFactory:
    """模型工厂类，用于加载不同类型的模型"""
    
    @staticmethod
    def load_model(model_type: str, model_name: str, device: str):
        """
        加载指定类型的模型
        Args:
            model_type: 模型类型 ('clip', 'openclip', 'siglip')
            model_name: 模型名称
            device: 计算设备
        Returns:
            模型、预处理器和分词器
        """
        if model_type.lower() == 'clip':
            # Hugging Face CLIP implementation
            model = CLIPModel.from_pretrained(model_name).to(device)
            processor = CLIPProcessor.from_pretrained(model_name)
            return model, processor, processor
            
        elif model_type.lower() == 'openclip':
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained='openai', device=device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            return model, preprocess, tokenizer
            
        elif model_type.lower() == 'siglip':
            model = AutoModel.from_pretrained(model_name).to(device)
            processor = AutoProcessor.from_pretrained("google/siglip2-so400m-patch14-384")
            return model, processor, processor
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class ReasoningClassificationEvaluator:
    def __init__(self, model_type="clip", model_name="ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化评估器
        Args:
            model_type: 模型类型 ('clip', 'openclip', 'siglip')
            model_name: 模型名称
            device: 计算设备
        """
        self.device = device
        self.model_type = model_type.lower()
        self.model_name = model_name
        
        # 加载模型
        self.model, self.preprocess, self.tokenizer = ModelFactory.load_model(
            model_type, model_name, device
        )
        
        print(f"Loaded {model_type.upper()} model: {model_name} on {device}")
    
    def _ensure_list(self, data):
        """确保数据是列表格式"""
        import numpy as np
        
        if isinstance(data, list):
            return data
        elif isinstance(data, np.ndarray):
            # numpy数组转换为列表
            return data.tolist()
        elif isinstance(data, str):
            try:
                import ast
                return ast.literal_eval(data)
            except:
                return [data]
        else:
            return [str(data)]
    
    def load_image(self, image_path: str, base_path: str = "/home/muzammal/Projects/CLIP-R/data/Xkev-LLaVA-CoT-100k") -> torch.Tensor:
        """
        加载并预处理图像
        Args:
            image_path: 相对图像路径
            base_path: 基础路径
        Returns:
            预处理后的图像张量
        """
        full_path = os.path.join(base_path, image_path)
        try:
            image = Image.open(full_path).convert('RGB')
            
            if self.model_type in ['siglip', 'clip']:
                # SigLIP和CLIP使用processor处理图像
                return image  # 返回PIL图像，后续在encode_image中处理
            else:
                # OpenCLIP使用preprocess
                return self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {full_path}: {e}")
            return None
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本列表
        Args:
            texts: 文本列表
        Returns:
            文本特征张量
        """
        with torch.no_grad():
            if self.model_type== 'clip':
                # SigLIP和CLIP使用processor处理文本
                inputs = self.tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
            elif self.model_type== 'siglip':
                inputs = self.tokenizer(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(self.device)
                text_features = self.model.get_text_features(**inputs)
            else:
                # OpenCLIP
                text_tokens = self.tokenizer(texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                
                # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def encode_image(self, image_input) -> torch.Tensor:
        """
        编码图像
        Args:
            image_input: 图像张量或PIL图像
        Returns:
            图像特征张量
        """
        with torch.no_grad():
            if self.model_type in ['siglip', 'clip']:
                # SigLIP和CLIP使用processor处理图像
                if isinstance(image_input, Image.Image):
                    inputs = self.preprocess(images=image_input, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                else:
                    raise ValueError(f"{self.model_type} requires PIL Image input")
            else:
                # OpenCLIP
                image_features = self.model.encode_image(image_input)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def compute_similarities(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算图像和文本之间的相似度
        Args:
            image_features: 图像特征
            text_features: 文本特征
        Returns:
            相似度分数
        """
        similarities = (image_features @ text_features.T).squeeze(0)
        return similarities
    
    def prepare_logic_val_options(self, row: pd.Series) -> Tuple[List[str], int]:
        """
        准备Logic Validation任务的选项：Logic Validation (Top-1)
        1 best_trp + 2 trp_neg
        Args:
            row: 数据行
        Returns:
            选项列表和正确答案索引
        """
        options = []
        
        # 添加best_trp (正确答案，应该排在第一)
        options.append(row['best_trp'])
        correct_idx = 0
        
        # 添加2个trp_neg
        trp_neg_list = self._ensure_list(row['trp_neg'])
        
        if len(trp_neg_list) >= 2:
            options.extend(trp_neg_list[:2])
        else:
            print(f"Warning: trp_neg has less than 2 items for row {row['id']}")
            options.extend(trp_neg_list)
        
        return options, correct_idx
    
    def prepare_best_reason_options(self, row: pd.Series) -> Tuple[List[str], int]:
        """
        准备Best-in-Class Reasoning任务的选项：Best-in-Class Reasoning (Top-1)
        1 best_trp + 2 trp (remove best_trp)
        Args:
            row: 数据行
        Returns:
            选项列表和正确答案索引
        """
        options = []
        
        # 添加best_trp (正确答案)
        options.append(row['best_trp'])
        correct_idx = 0
        
        # 添加2个trp (排除best_trp)
        trp_list = self._ensure_list(row['trp'])
        
        # 过滤掉best_trp
        filtered_trp = [trp for trp in trp_list if trp != row['best_trp']]
        if len(filtered_trp) >= 2:
            options.extend(filtered_trp[:2])
        else:
            # 如果过滤后不够2个，就用原始的trp
            options.extend(trp_list[:2])
        
        return options, correct_idx
    
    def prepare_reason_id_options(self, row: pd.Series) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        准备Reasoning Identification任务的选项：Reasoning Identification (Top-k)
        1 best_trp + 2 trp + 3 tb
        期望相似度: best_trp ~ tb > trp
        Args:
            row: 数据行
        Returns:
            选项列表和类别索引字典
        """
        options = []
        categories = {"best_trp": [], "trp": [], "tb": []}
        
        # 添加best_trp
        options.append(row['best_trp'])
        categories["best_trp"].append(0)
        
        # 添加2个trp (排除best_trp)
        trp_list = self._ensure_list(row['trp'])
        filtered_trp = [trp for trp in trp_list if trp != row['best_trp']]
        for i, trp in enumerate(filtered_trp[:2]):
            options.append(trp)
            categories["trp"].append(len(options) - 1)
        
        # 添加3个tb
        tb_list = self._ensure_list(row['tb'])
        for i, tb in enumerate(tb_list[:3]):
            options.append(tb)
            categories["tb"].append(len(options) - 1)
        
        return options, categories
    
    def evaluate_logic_val(self, df: pd.DataFrame, base_path: str) -> Dict:
        """
        评估Logic Validation任务：Logic Validation (Top-1)
        """
        print("Evaluating Logic Validation (Top-1)")
        correct_predictions = 0
        total_samples = 0
        failed_samples = 0
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Logic Val"):
            # 准备选项
            options, correct_idx = self.prepare_logic_val_options(row)
            
            # 加载图像
            image_input = self.load_image(row['image_path'], base_path)
            if image_input is None:
                failed_samples += 1
                continue
            
            # 编码图像和文本
            image_features = self.encode_image(image_input)
            text_features = self.encode_texts(options)
            
            # 计算相似度
            similarities = self.compute_similarities(image_features, text_features)
            predicted_idx = torch.argmax(similarities).item()
            
            # 记录结果
            is_correct = predicted_idx == correct_idx
            correct_predictions += is_correct
            total_samples += 1
            
            results.append({
                'id': row['id'],
                'image_path': row['image_path'],
                'options': options,
                'similarities': similarities.cpu().tolist(),
                'predicted_idx': predicted_idx,
                'correct_idx': correct_idx,
                'is_correct': is_correct
            })
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return {
            'task': 'Logic Validation (Top-1)',
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_samples': total_samples,
            'failed_samples': failed_samples,
            'results': results
        }
    
    def evaluate_best_reason(self, df: pd.DataFrame, base_path: str) -> Dict:
        """
        评估Best-in-Class Reasoning任务：Best-in-Class Reasoning (Top-1)
        """
        print("Evaluating Best-in-Class Reasoning (Top-1)")
        correct_predictions = 0
        total_samples = 0
        failed_samples = 0
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Best Reason"):
            # 准备选项
            options, correct_idx = self.prepare_best_reason_options(row)
            
            # 加载图像
            image_input = self.load_image(row['image_path'], base_path)
            if image_input is None:
                failed_samples += 1
                continue
            
            # 编码图像和文本
            image_features = self.encode_image(image_input)
            text_features = self.encode_texts(options)
            
            # 计算相似度
            similarities = self.compute_similarities(image_features, text_features)
            predicted_idx = torch.argmax(similarities).item()
            
            # 记录结果
            is_correct = predicted_idx == correct_idx
            correct_predictions += is_correct
            total_samples += 1
            
            results.append({
                'id': row['id'],
                'image_path': row['image_path'],
                'options': options,
                'similarities': similarities.cpu().tolist(),
                'predicted_idx': predicted_idx,
                'correct_idx': correct_idx,
                'is_correct': is_correct
            })
        
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        
        return {
            'task': 'Best-in-Class Reasoning (Top-1)',
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_samples': total_samples,
            'failed_samples': failed_samples,
            'results': results
        }
    
    def evaluate_reason_id(self, df: pd.DataFrame, base_path: str) -> Dict:
        """
        评估Reasoning Identification任务：Reasoning Identification (Top-k)
        期望相似度: best_trp ~ tb > trp
        """
        print("Evaluating Reasoning Identification (Top-k)")
        total_samples = 0
        failed_samples = 0
        
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Reason ID"):
            # 准备选项
            options, categories = self.prepare_reason_id_options(row)
            
            # 加载图像
            image_input = self.load_image(row['image_path'], base_path)
            if image_input is None:
                failed_samples += 1
                continue
            
            # 编码图像和文本
            image_features = self.encode_image(image_input)
            text_features = self.encode_texts(options)
            
            # 计算相似度
            similarities = self.compute_similarities(image_features, text_features)
            
            # 计算各类别的平均相似度
            avg_similarities = {}
            for category, indices in categories.items():
                if indices:
                    avg_similarities[category] = similarities[indices].mean().item()
                else:
                    avg_similarities[category] = 0.0
            
            # 计算排序（从高到低）
            sorted_indices = torch.argsort(similarities, descending=True).cpu().tolist()
            
            # Top-k准确率分析：检查推理选项（best_trp + trp）是否在Top-2中
            reasoning_indices = categories['best_trp'] + categories['trp']  # 所有推理选项的索引
            top2_indices = sorted_indices[:2]  # 前两名的索引
            top2_correct = any(idx in reasoning_indices for idx in top2_indices)
            
            # 排序分析：计算推理选项和基础描述选项的平均排名
            reasoning_ranks = []  # 推理选项的排名
            tb_ranks = []  # 基础描述选项的排名
            
            for i, idx in enumerate(sorted_indices):
                rank = i + 1  # 排名从1开始
                if idx in reasoning_indices:
                    reasoning_ranks.append(rank)
                elif idx in categories['tb']:
                    tb_ranks.append(rank)
            
            mean_reasoning_rank = sum(reasoning_ranks) / len(reasoning_ranks) if reasoning_ranks else 0
            mean_tb_rank = sum(tb_ranks) / len(tb_ranks) if tb_ranks else 0
            
            total_samples += 1
            
            results.append({
                'id': row['id'],
                'image_path': row['image_path'],
                'options': options,
                'categories': categories,
                'similarities': similarities.cpu().tolist(),
                'avg_similarities': avg_similarities,
                'sorted_indices': sorted_indices,
                'top2_correct': top2_correct,
                'reasoning_ranks': reasoning_ranks,
                'tb_ranks': tb_ranks,
                'mean_reasoning_rank': mean_reasoning_rank,
                'mean_tb_rank': mean_tb_rank
            })
        
        # 计算整体统计指标
        if results:
            # Top-2准确率
            top2_correct_count = sum(1 for r in results if r['top2_correct'])
            top2_accuracy = top2_correct_count / len(results)
            
            # 平均排名分析
            all_reasoning_ranks = []
            all_tb_ranks = []
            for r in results:
                all_reasoning_ranks.extend(r['reasoning_ranks'])
                all_tb_ranks.extend(r['tb_ranks'])
            
            overall_mean_reasoning_rank = sum(all_reasoning_ranks) / len(all_reasoning_ranks) if all_reasoning_ranks else 0
            overall_mean_tb_rank = sum(all_tb_ranks) / len(all_tb_ranks) if all_tb_ranks else 0
            
            # 排名差异（越大越好，表示推理选项排名更靠前）
            rank_difference = overall_mean_tb_rank - overall_mean_reasoning_rank
        else:
            top2_accuracy = 0
            overall_mean_reasoning_rank = 0
            overall_mean_tb_rank = 0
            rank_difference = 0
        
        return {
            'task': 'Reasoning Identification (Top-k)',
            'total_samples': total_samples,
            'failed_samples': failed_samples,
            'top2_accuracy': top2_accuracy,
            'top2_correct_count': top2_correct_count if results else 0,
            'overall_mean_reasoning_rank': overall_mean_reasoning_rank,
            'overall_mean_tb_rank': overall_mean_tb_rank,
            'rank_difference': rank_difference,
            'results': results
        }
    
    def run_single_task(self, task_name: str, data_path: str, base_path: str, output_dir: str = None):
        """
        运行单个任务评估
        Args:
            task_name: 任务名称 ('logic_val', 'best_reason', 'reason_id')
            data_path: 数据文件路径
            base_path: 图像基础路径
            output_dir: 输出目录
        """
        print(f"🚀 Starting evaluation with {self.model_type} model: {self.model_name}")
        print(f"📝 Task: {task_name}")
        print(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df)} samples")
        
        # 根据任务名称运行对应的评估
        if task_name == 'logic_val':
            results = self.evaluate_logic_val(df, base_path)
            print(f"\n✅ Logic Validation - {results['task']}:")
            print(f"   🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"   ✓ Correct: {results['correct_predictions']}/{results['total_samples']}")
            print(f"   ❌ Failed: {results['failed_samples']}")
            
        elif task_name == 'best_reason':
            results = self.evaluate_best_reason(df, base_path)
            print(f"\n✅ Best Reasoning - {results['task']}:")
            print(f"   🎯 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
            print(f"   ✓ Correct: {results['correct_predictions']}/{results['total_samples']}")
            print(f"   ❌ Failed: {results['failed_samples']}")
            
        elif task_name == 'reason_id':
            results = self.evaluate_reason_id(df, base_path)
            print(f"\n✅ Reasoning ID - {results['task']}:")
            print(f"   📊 Total samples: {results['total_samples']}")
            print(f"   ❌ Failed: {results['failed_samples']}")
            print(f"   🎯 Top-2 Accuracy: {results['top2_accuracy']:.4f} ({results['top2_accuracy']*100:.2f}%)")
            print(f"   ✓ Top-2 Correct: {results['top2_correct_count']}/{results['total_samples'] - results['failed_samples']}")
            print(f"   📈 Ranking Analysis:")
            print(f"      • Mean Reasoning Rank: {results['overall_mean_reasoning_rank']:.2f}")
            print(f"      • Mean TB Rank: {results['overall_mean_tb_rank']:.2f}")
            print(f"      • Rank Difference: {results['rank_difference']:.2f} (higher is better)")
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            model_safe_name = f"{self.model_type}_{self.model_name.replace('/', '_').replace('-', '_')}"
            
            # 所有任务都分开保存：总结结果和详细记录
            
            # 1. 保存总结结果（不包含详细的每条记录）
            summary_results = {k: v for k, v in results.items() if k != 'results'}
            summary_file = os.path.join(output_dir, f'{task_name}_summary_{model_safe_name}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary_results, f, indent=2)
            
            # 2. 保存详细记录
            detailed_results = {
                'task': results['task'],
                'total_samples': results['total_samples'],
                'failed_samples': results['failed_samples'],
                'results': results['results']
            }
            detailed_file = os.path.join(output_dir, f'{task_name}_detailed_{model_safe_name}.json')
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"\n💾 Results saved to:")
            print(f"   📊 Summary: {summary_file}")
            print(f"   📝 Detailed: {detailed_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Zero-Shot Reasoning Classification')
    parser.add_argument('--data_path', type=str, 
                       default='/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItw/itw_final_with_options.parquet',
                       help='Path to the cleaned data file')
    parser.add_argument('--base_path', type=str,
                       default='/home/muzammal/Projects/CLIP-R/data/fesvhtr-CLIPReasonItwImages/llavacot_test_images',
                       help='Base path for images')
    parser.add_argument('--model_type', type=str, default='siglip',
                       choices=['clip', 'openclip', 'siglip'],
                       help='Type of model to use')
    parser.add_argument('--model_name', type=str, default="fesvhtr/siglip-r-s1-run1027-1926",
                       help='Model name/path to use')
    parser.add_argument('--output_dir', type=str, default='/home/muzammal/Projects/CLIP-R/eval/results_reasonpro',
                       help='Output directory for results')
    parser.add_argument('--task', type=str, required=True,
                       choices=['logic_val', 'best_reason', 'reason_id'],
                       help='Which task to run')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    

    
    # 创建评估器
    evaluator = ReasoningClassificationEvaluator(
        model_type=args.model_type, 
        model_name=args.model_name, 
        device=device
    )
    
    # 运行单个任务评估
    results = evaluator.run_single_task(
        task_name=args.task,
        data_path=args.data_path,
        base_path=args.base_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
