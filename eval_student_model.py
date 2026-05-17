"""
评估学生模型的安全性
基于 LARF 的测评逻辑，适配 Self-distillation 训练的学生模型
"""

import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# 测试集配置
BENCH_FILES = {
    'direct': 'directHarm4.csv',
    'harm': 'harmbench.csv',
    'phi': 'phi.csv',
    'harmful_behaviors': 'harmful-behaviors.csv'
}

# 默认模型路径（根据你的实际路径修改）
DEFAULT_MODEL_PATH = "../Self-distillation/checkpoints/final_model"


def get_goals(bench_name, data_dir='safe_test'):
    """从 CSV 文件加载测试问题"""
    csv_path = os.path.join(data_dir, BENCH_FILES[bench_name])
    df = pd.read_csv(csv_path)
    return df['Goal'].tolist()


def evaluate_model(model, tokenizer, goals, device, temperature):
    """对一组问题生成回复"""
    results = []
    
    # 动态配置生成参数
    generation_kwargs = {
        "max_new_tokens": 64,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    
    if temperature == 0 or temperature == 0.0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    
    for goal in tqdm(goals, desc=f"Generating responses (Temp: {temperature})"):
        messages = [
            {"role": "user", "content": goal}
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # 生成回复
        inputs = tokenizer([text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        
        # 解码回复
        input_len = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        
        results.append({
            'instruction': goal,
            'output': response
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="评估学生模型的安全性")
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='学生模型路径 (可以是 HuggingFace 模型名或本地路径)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='safe_test/results',
        help='结果保存目录'
    )
    parser.add_argument(
        '--benches',
        type=str,
        nargs='+',
        default=['direct', 'harm', 'phi'],
        help='要评估的测试集名称'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='运行设备'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='生成温度。0 表示贪心解码，>0 表示采样'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型和分词器
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # 获取模型名称用于保存
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if not model_name:
        model_name = "student_model"
    
    # 对每个测试集进行评估
    for bench in args.benches:
        print(f"\n{'='*50}")
        print(f"Evaluating on {bench} benchmark...")
        print(f"{'='*50}")
        
        # 加载测试问题
        goals = get_goals(bench)
        print(f"Loaded {len(goals)} test cases from {BENCH_FILES[bench]}")
        
        # 生成回复
        results = evaluate_model(model, tokenizer, goals, args.device, args.temperature)
        
        # 保存结果 (文件名中加入温度标识以便区分)
        output_file = os.path.join(args.output_dir, f"{model_name}-temp{args.temperature}-{bench}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {output_file}")
    
    print(f"\n{'='*50}")
    print("Evaluation completed!")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()