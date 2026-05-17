"""
评估学生模型的安全性 — 数据集来自 LARF/datasets/ 的 csv (Goal 列)

与 eval_student_model.py 的区别:
  - 默认数据目录 ./datasets (而不是 ./safe_test)
  - --benches 接受任意 csv 文件名 / 相对路径 / 绝对路径, 不再限制于固定的 4 个 key
  - 仍兼容旧的 4 个短 key (direct/harm/phi/harmful_behaviors), 自动映射到 safe_test/ 下原来的文件名

输出 json 文件名: <model_name>-temp<temp>-<bench_stem>.json
其中 bench_stem 是 csv 文件名去掉 .csv 后缀。
"""

import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# 旧 4 个 bench 的兼容映射: <短 key> -> <safe_test 下的 csv 文件名>
LEGACY_BENCH_FILES = {
    'direct': ('safe_test', 'directHarm4.csv'),
    'harm': ('safe_test', 'harmbench.csv'),
    'phi': ('safe_test', 'phi.csv'),
    'harmful_behaviors': ('safe_test', 'harmful-behaviors.csv'),
}

DEFAULT_MODEL_PATH = "../Self-distillation/checkpoints/final_model"


def resolve_bench(bench_arg, data_dir):
    """把一个 --benches 参数解析成 (bench_stem, csv_path).

    优先级:
      1. 绝对路径 (以 / 开头)                  -> 直接用
      2. 相对路径含 /                           -> 相对当前 cwd 解析
      3. 文件名 (含 .csv) 在 data_dir 下存在     -> data_dir/<name>
      4. 文件名 (不含 .csv) 在 data_dir 下存在  -> data_dir/<name>.csv
      5. 是 LEGACY_BENCH_FILES 的 key (兼容老脚本) -> safe_test/<旧文件名>
    """
    # 1. 绝对路径
    if os.path.isabs(bench_arg):
        if not os.path.isfile(bench_arg):
            raise FileNotFoundError(f"--benches 绝对路径不存在: {bench_arg}")
        stem = os.path.splitext(os.path.basename(bench_arg))[0]
        return stem, bench_arg

    # 2. 相对路径含 /
    if '/' in bench_arg:
        if not os.path.isfile(bench_arg):
            raise FileNotFoundError(f"--benches 相对路径不存在: {bench_arg}")
        stem = os.path.splitext(os.path.basename(bench_arg))[0]
        return stem, bench_arg

    # 3 / 4. 在 data_dir 下找
    candidates = []
    if bench_arg.lower().endswith('.csv'):
        candidates.append(os.path.join(data_dir, bench_arg))
    else:
        candidates.append(os.path.join(data_dir, bench_arg + '.csv'))
        candidates.append(os.path.join(data_dir, bench_arg))  # 万一文件本身没后缀
    for cand in candidates:
        if os.path.isfile(cand):
            stem = os.path.splitext(os.path.basename(cand))[0]
            return stem, cand

    # 5. legacy 短 key
    if bench_arg in LEGACY_BENCH_FILES:
        legacy_dir, legacy_file = LEGACY_BENCH_FILES[bench_arg]
        legacy_path = os.path.join(legacy_dir, legacy_file)
        if os.path.isfile(legacy_path):
            return bench_arg, legacy_path  # 用短 key 当 stem 保持向后兼容

    raise FileNotFoundError(
        f"--benches '{bench_arg}' 解析失败; "
        f"既不是绝对/相对路径, 也没在 {data_dir}/ 下找到对应 csv, "
        f"也不是 legacy key ({list(LEGACY_BENCH_FILES.keys())})"
    )


def get_goals(csv_path):
    """从 csv 读 Goal 列; 对未转义逗号等脏数据自动 fallback 到宽容解析"""
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"[warn] 默认 c-parser 失败 ({e}); 改用 python 引擎 + skip bad lines")
        df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    if 'Goal' not in df.columns:
        raise KeyError(
            f"{csv_path} 缺少 'Goal' 列 (实际列: {list(df.columns)})"
        )
    return df['Goal'].dropna().tolist()


def _render_prompt(tokenizer, goal):
    messages = [{"role": "user", "content": goal}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def evaluate_model(model, tokenizer, goals, device, temperature, batch_size=32):
    """对一组问题批量生成回复 (left-padded so all rows share input_len)。"""
    results = []

    generation_kwargs = {
        "max_new_tokens": 64,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature == 0 or temperature == 0.0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature

    # Decoder-only LM 必须左 padding, 否则 attention_mask 之后的生成会从 pad 起
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        n = len(goals)
        for i in tqdm(range(0, n, batch_size),
                      desc=f"Generating (Temp: {temperature}, bs={batch_size})"):
            batch_goals = goals[i:i + batch_size]
            texts = [_render_prompt(tokenizer, g) for g in batch_goals]
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)
            input_len = inputs.input_ids.shape[1]
            for goal, row in zip(batch_goals, outputs):
                response = tokenizer.decode(row[input_len:], skip_special_tokens=True)
                results.append({'instruction': goal, 'output': response})
    finally:
        tokenizer.padding_side = orig_padding_side

    return results


def main():
    parser = argparse.ArgumentParser(description="评估学生模型的安全性 (datasets/ 版本)")
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='学生模型路径 (HF 名或本地路径)')
    parser.add_argument('--output_dir', type=str, default='safe_test/results',
                        help='结果保存目录')
    parser.add_argument('--data_dir', type=str, default='datasets',
                        help='测试集 csv 根目录, 默认 ./datasets')
    parser.add_argument('--benches', type=str, nargs='+', required=True,
                        help='测试集列表; 每项可以是: csv 文件名 (不含路径, 在 data_dir 里找), '
                             '相对/绝对 csv 路径, 或 legacy 短 key '
                             '(direct/harm/phi/harmful_behaviors -> safe_test/)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='生成温度。0 表示贪心解码, >0 表示采样')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='单次 model.generate 处理的样本数 (left-padded). 越大越快, 受显存约束。')
    args = parser.parse_args()

    # 先把所有 bench 解析掉, 早失败胜过跑了一半才崩
    bench_specs = [resolve_bench(b, args.data_dir) for b in args.benches]
    print("待评测 bench:")
    for stem, path in bench_specs:
        print(f"  - {stem:30s} <- {path}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if args.device == 'cuda' else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    model_name = os.path.basename(args.model_path.rstrip('/')) or "student_model"

    for stem, csv_path in bench_specs:
        print(f"\n{'='*50}\nEvaluating on {stem} ({csv_path})\n{'='*50}")
        goals = get_goals(csv_path)
        print(f"Loaded {len(goals)} test cases")

        results = evaluate_model(model, tokenizer, goals, args.device, args.temperature,
                                 batch_size=args.batch_size)

        out_file = os.path.join(args.output_dir, f"{model_name}-temp{args.temperature}-{stem}.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {out_file}")

    print(f"\n{'='*50}\nEvaluation completed!\n{'='*50}")


if __name__ == '__main__':
    main()
