#!/bin/bash
# ==============================================================================
# run_models_overrefusal.sh — 对一组(已存在的) HF 模型做 over-refusal 评测
#   评测集: XSTest (safe/unsafe) + OR-Bench (hard-1k / toxic)
#   判官:   关键词/子串匹配 (refusal_judge.py, 纯 CPU, 无需 vLLM)
#
#   与 run_models.sh 的关系:
#     run_models.sh:            有害 benchmark + Llama-Guard 打分 -> ASR (越低越安全)
#     run_models_overrefusal.sh: 过度拒绝 benchmark + 关键词判官  -> refusal_rate
#                                安全集上越低越好(过度拒绝少), 不安全集上越高越好(正确拒绝)
#
#   用法:
#     bash run_models_overrefusal.sh <model_path_or_id> [<model_path_or_id> ...]
#     MODELS="m1 m2 m3" bash run_models_overrefusal.sh
#     bash run_models_overrefusal.sh -f models.txt           # 从文件读, 每行一个模型
#
#   首次使用前请先下载数据集:
#     python download_overrefusal_datasets.py --out_dir datasets
#
#   输出:
#     ${OUTPUT_ROOT}/<basename(model)>/${EVAL_SUBDIR}/{temp_0[,temp_1...]}
#     ${OUTPUT_ROOT}/overrefusal_summary_${DATE_TAG}.{csv,json}
#
#   关键环境变量:
#     OUTPUT_ROOT        输出根目录
#     BATCH_SIZE         并行模型数 (= 用几张卡)              默认 8
#     EVAL_BATCH_SIZE    单卡 model.generate 的样本数         默认 32
#     GEN_MAX_NEW_TOKENS 每条回复生成长度 (越长合规信号越足)  默认 256
#     N_TEMP1_RUNS       额外的 temperature=1 采样轮数        默认 0 (只跑贪心 temp_0)
#     DATA_DIR           数据集 csv 目录                       默认 datasets
#     LARF_DIR           代码目录                              默认 = 脚本所在目录
# ==============================================================================
set +e

# 默认用脚本自身所在目录作为代码根 (比 run_models.sh 写死 /opt/tiger/entry 更可移植)
LARF_DIR="${LARF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="overrefusal_${DATE_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${LARF_DIR}/_overrefusal_eval_${DATE_TAG}}"

BATCH_SIZE="${BATCH_SIZE:-8}"               # 并行模型数 (跨 GPU)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"    # 单卡 model.generate 的样本数
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-256}"
N_TEMP1_RUNS="${N_TEMP1_RUNS:-0}"           # 额外 temp=1 采样轮数 (0=只跑 temp_0 贪心)
DATA_DIR="${DATA_DIR:-datasets}"
SCORE_FIELD="${SCORE_FIELD:-score_refusal}"

# over-refusal 评测集 (对应 datasets/<name>.csv)
BENCHES=(xstest_safe xstest_unsafe or_bench_hard_1k or_bench_toxic)

# ------------- 解析模型列表 (与 run_models.sh 完全一致) -------------
MODELS=()
if [ "$1" = "-f" ] && [ -n "$2" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line//$'\r'/}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] && MODELS+=("$line")
    done < "$2"
elif [ "$#" -gt 0 ]; then
    MODELS=("$@")
elif [ -n "${MODELS_ENV:-${MODELS:-}}" ]; then
    read -r -a MODELS <<< "${MODELS_ENV:-${MODELS}}"
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[run_overrefusal] 用法: bash $0 <model_path> [<model_path> ...]"
    echo "                  或: MODELS=\"m1 m2\" bash $0"
    echo "                  或: bash $0 -f models.txt"
    exit 1
fi

# ------------- 数据集存在性检查 -------------
MISSING=()
for b in "${BENCHES[@]}"; do
    [ -f "${LARF_DIR}/${DATA_DIR}/${b}.csv" ] || MISSING+=("${b}.csv")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    echo "[error] 缺少数据集: ${MISSING[*]}"
    echo "        先运行: cd ${LARF_DIR} && python download_overrefusal_datasets.py --out_dir ${DATA_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[run_overrefusal] 共 ${#MODELS[@]} 个模型, 输出根目录: ${OUTPUT_ROOT}"
echo "[run_overrefusal] benches: ${BENCHES[*]}"
echo "[run_overrefusal] temp_0 (贪心) + ${N_TEMP1_RUNS} 轮 temp_1; max_new_tokens=${GEN_MAX_NEW_TOKENS}"
for m in "${MODELS[@]}"; do echo "  - $m"; done

cd "${LARF_DIR}"

EVAL_BASES=()

# ============ 阶段 1: 推理 ============
for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_ID=$(( i % BATCH_SIZE ))
    _MP="${MODEL_PATH%/}"
    NAME=$(basename "${_MP}")
    if [[ "${NAME}" == checkpoint-* ]]; then
        PARENT=$(basename "$(dirname "${_MP}")")
        NAME="${PARENT}-${NAME}"
    fi

    OUT_BASE="${OUTPUT_ROOT}/${NAME}/${EVAL_SUBDIR}"
    EVAL_BASES+=("${OUT_BASE}")
    mkdir -p "${OUT_BASE}/logs"
    LOG="${OUT_BASE}/logs/eval.log"
    echo ">>> [GPU ${GPU_ID}] ${NAME} → ${MODEL_PATH}"
    (
        CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model_usedatasets.py \
            --model_path "${MODEL_PATH}" --output_dir "${OUT_BASE}/temp_0" \
            --data_dir "${DATA_DIR}" --batch_size ${EVAL_BATCH_SIZE} \
            --max_new_tokens ${GEN_MAX_NEW_TOKENS} \
            --temperature 0 --benches "${BENCHES[@]}"
        for r in $(seq 1 "${N_TEMP1_RUNS}"); do
            sub="temp_1"; [ "$r" -gt 1 ] && sub="temp_1_run${r}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model_usedatasets.py \
                --model_path "${MODEL_PATH}" --output_dir "${OUT_BASE}/${sub}" \
                --data_dir "${DATA_DIR}" --batch_size ${EVAL_BATCH_SIZE} \
                --max_new_tokens ${GEN_MAX_NEW_TOKENS} \
                --temperature 1 --benches "${BENCHES[@]}"
        done
    ) > "${LOG}" 2>&1 &

    (( (i + 1) % BATCH_SIZE == 0 )) && wait
done
wait
echo "[run_overrefusal] 阶段 1 (推理) 完成"

# ============ 阶段 2: 关键词拒绝判官 (纯 CPU, 无需 vLLM) ============
for TARGET in "${EVAL_BASES[@]}"; do
    [ -d "${TARGET}" ] || continue
    mkdir -p "${TARGET}/logs"
    echo ">>> 判官打分: ${TARGET}"
    python "${LARF_DIR}/refusal_judge.py" \
        --input_dir   "${TARGET}" \
        --score_field "${SCORE_FIELD}" \
        2>&1 | tee "${TARGET}/logs/refusal_judge.log"
done
echo "[run_overrefusal] 阶段 2 (判官) 完成"

# ============ 阶段 3: 汇总 over-refusal 指标 ============
SUMMARY_CSV="${OUTPUT_ROOT}/overrefusal_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/overrefusal_summary_${DATE_TAG}.json"

PARENTS=()
for TARGET in "${EVAL_BASES[@]}"; do
    PARENTS+=("$(dirname "${TARGET}")")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${SCORE_FIELD}" "${PARENTS[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, eval_subdir, score_field, *model_paths = sys.argv[1:]


def bench_kind(stem):
    """safe: 安全集, refusal=过度拒绝(越低越好); unsafe: 有害集, refusal=正确拒绝(越高越好)"""
    s = stem.lower()
    if "unsafe" in s or "toxic" in s:
        return "unsafe"
    if "safe" in s or "hard" in s:
        return "safe"
    return "unknown"


rows = []
for mp in model_paths:
    name = os.path.basename(mp.rstrip('/'))
    eval_dir = os.path.join(mp, eval_subdir)
    if not os.path.isdir(eval_dir):
        continue
    for root, _, files in os.walk(eval_dir):
        for fn in files:
            if not fn.endswith('.json'):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[summary] WARN: 读取失败 {fp}: {e}")
                continue
            if not isinstance(data, list) or not data:
                continue
            scored = [it for it in data
                      if isinstance(it, dict) and it.get(score_field) in (0, 1)]
            if not scored:
                continue
            refused = sum(1 for it in scored if int(it[score_field]) == 1)
            total = len(scored)
            stem = os.path.splitext(fn)[0]
            # bench stem 形如 <model>-temp<T>-<benchname>; 取最后一段作 bench 名
            parts = stem.split('-temp')
            benchname = parts[-1].split('-', 1)[-1] if len(parts) > 1 else stem
            kind = bench_kind(benchname)
            refusal_rate = refused / total if total else 0.0
            rows.append({
                'model': name,
                'split': os.path.relpath(root, eval_dir) if os.path.relpath(root, eval_dir) != '.' else '',
                'bench': benchname,
                'kind': kind,
                'refused': refused,
                'total': total,
                'refusal_rate': refusal_rate,
                'compliance_rate': 1.0 - refusal_rate,
                # 头条指标: safe 集看过度拒绝率, unsafe 集看正确拒绝率
                'headline': refusal_rate if kind == 'safe' else (1.0 - refusal_rate if kind == 'unsafe' else refusal_rate),
                'headline_name': 'over_refusal_rate' if kind == 'safe' else ('over_compliance_rate' if kind == 'unsafe' else 'refusal_rate'),
            })

rows.sort(key=lambda r: (r['model'], r['split'], r['kind'], r['bench']))

cols = ['model', 'split', 'bench', 'kind', 'refused', 'total',
        'refusal_rate', 'compliance_rate', 'headline_name', 'headline']
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(cols)
    for r in rows:
        w.writerow([r['model'], r['split'], r['bench'], r['kind'], r['refused'], r['total'],
                    f"{r['refusal_rate']:.4f}", f"{r['compliance_rate']:.4f}",
                    r['headline_name'], f"{r['headline']:.4f}"])

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n========== Over-Refusal Summary ==========")
print("safe 集: over_refusal_rate 越低越好 | unsafe 集: over_compliance_rate 越低越好\n")
print(f"{'model':<45} {'split':<10} {'bench':<18} {'kind':<8} {'refused':>7} {'total':>6} "
      f"{'refusal':>8} {'headline':>22}")
for r in rows:
    print(f"{r['model']:<45} {r['split']:<10} {r['bench']:<18} {r['kind']:<8} "
          f"{r['refused']:>7} {r['total']:>6} {r['refusal_rate']:>8.4f} "
          f"{r['headline_name']+'='+format(r['headline'],'.4f'):>22}")
print(f"\n[summary] CSV  → {csv_path}")
print(f"[summary] JSON → {json_path}")
PY

echo "[run_overrefusal] 全部完成! 结果在 ${OUTPUT_ROOT}/<model>/${EVAL_SUBDIR}/"
