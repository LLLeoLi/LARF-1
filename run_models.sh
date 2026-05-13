#!/bin/bash
# ==============================================================================
# run_models.sh — 对一组(已存在的) HF 模型直接做 benchmark 评测
#   与 run.sh 流程一致, 但模型来源不同:
#     run.sh:        从 ${CKPT_ROOT} 下按 ${CKPT_GLOB} 检索 checkpoint 目录
#     run_models.sh: 通过命令行参数 / MODELS 环境变量 直接给定一组模型路径
#
#   用法:
#     bash run_models.sh <model_path_or_id> [<model_path_or_id> ...]
#     MODELS="m1 m2 m3" bash run_models.sh
#     bash run_models.sh -f models.txt           # 从文件读, 每行一个模型
#
#   输出:
#     ${OUTPUT_ROOT}/<basename(model)>/${EVAL_SUBDIR}/{temp_0,temp_1,temp_1_run2,temp_1_run3}
#     ${OUTPUT_ROOT}/asr_summary_${DATE_TAG}.{csv,json}
# ==============================================================================
set +e

ENTRY_DIR="/opt/tiger/entry"
LARF_DIR="${ENTRY_DIR}/LARF"
LLAMA_GUARD_MODEL="${ENTRY_DIR}/Llama-Guard-4-12B"

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="eval_${DATE_TAG}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/_models_eval_${DATE_TAG}}"
VLLM_LOG="${LARF_DIR}/_vllm_${DATE_TAG}.log"

BATCH_SIZE="${BATCH_SIZE:-8}"             # 并行模型数 (跨 GPU)
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"  # 单卡 model.generate 的样本数
DATA_DIR="${DATA_DIR:-datasets}"
BENCHES=(advbench ALERT HarmfulQA JBB-Behaviors PKU-SafeRLHF-30K sorry_bench_202503 harmbench)

# ------------- 解析模型列表 -------------
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
    # 兼容: MODELS="m1 m2" bash run_models.sh
    read -r -a MODELS <<< "${MODELS_ENV:-${MODELS}}"
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[run_models] 用法: bash $0 <model_path> [<model_path> ...]"
    echo "             或: MODELS=\"m1 m2\" bash $0"
    echo "             或: bash $0 -f models.txt"
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
echo "[run_models] 共 ${#MODELS[@]} 个模型, 输出根目录: ${OUTPUT_ROOT}"
for m in "${MODELS[@]}"; do echo "  - $m"; done

cd "${LARF_DIR}"

# 记录 (model_path, out_base) 对, 阶段 3/4 复用
EVAL_BASES=()

# ============ 阶段 1: 推理 ============
for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_ID=$(( i % BATCH_SIZE ))
    _MP="${MODEL_PATH%/}"
    NAME=$(basename "${_MP}")
    # 若 basename 形如 checkpoint-XXX, 拼上上一级目录名以避免不同模型同名冲突
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
            --temperature 0 --benches "${BENCHES[@]}"
        for r in 1 2 3; do
            sub="temp_1"; [ "$r" -gt 1 ] && sub="temp_1_run${r}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model_usedatasets.py \
                --model_path "${MODEL_PATH}" --output_dir "${OUT_BASE}/${sub}" \
                --data_dir "${DATA_DIR}" --batch_size ${EVAL_BATCH_SIZE} \
                --temperature 1 --benches "${BENCHES[@]}"
        done
    ) > "${LOG}" 2>&1 &

    (( (i + 1) % BATCH_SIZE == 0 )) && wait
done
wait
echo "[run_models] 阶段 1 完成"

# ============ 阶段 2: vLLM Llama-Guard 服务 (port 8000) ============
vllm serve --model "${LLAMA_GUARD_MODEL}" --port=8000  --max-model-len 1000 \
    --data-parallel-size=8 --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

if [ -z "${VLLM_PID}" ] || ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[error] vLLM 启动失败, 日志: ${VLLM_LOG}"
    exit 1
fi

cleanup_vllm() {
    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[cleanup] 终止 vLLM (PID=${VLLM_PID}) ..."
        kill -TERM "${VLLM_PID}" 2>/dev/null
        sleep 2
        kill -0 "${VLLM_PID}" 2>/dev/null && kill -9 "${VLLM_PID}" 2>/dev/null
        wait "${VLLM_PID}" 2>/dev/null
    fi
}
trap cleanup_vllm EXIT INT TERM

echo "[run_models] 等待 vLLM (PID=${VLLM_PID}) 就绪 ..."
VLLM_READY=false
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[run_models] vLLM 就绪 (尝试 ${i} 次)"
        VLLM_READY=true
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[error] vLLM 进程已退出, 日志: ${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done

if [ "${VLLM_READY}" != "true" ]; then
    echo "[error] vLLM 在 600 秒内未就绪, 日志: ${VLLM_LOG}"
    exit 1
fi

# ============ 阶段 3: Llama Guard 打分 ============
for TARGET in "${EVAL_BASES[@]}"; do
    [ -d "${TARGET}" ] || continue

    if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[error] vLLM 服务在打分前已不可达, 日志: ${VLLM_LOG}"
        exit 1
    fi

    mkdir -p "${TARGET}/logs"
    echo ">>> 打分: ${TARGET}"
    python "${LARF_DIR}/llama_guard.py" --input_dir "${TARGET}" 2>&1 | tee "${TARGET}/logs/guard.log"
done

# ============ 阶段 4: 汇总 ASR ============
SUMMARY_CSV="${OUTPUT_ROOT}/asr_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${OUTPUT_ROOT}/asr_summary_${DATE_TAG}.json"

# 这里把 "<model>/<EVAL_SUBDIR>" 的父目录传给 python, 复用 run.sh 的汇总逻辑
PARENTS=()
for TARGET in "${EVAL_BASES[@]}"; do
    PARENTS+=("$(dirname "${TARGET}")")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${PARENTS[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, eval_subdir, *model_paths = sys.argv[1:]
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
                print(f"[asr] WARN: 读取失败 {fp}: {e}")
                continue
            if not isinstance(data, list) or not data:
                continue
            scored = [it for it in data if isinstance(it, dict) and 'score' in it]
            if not scored:
                continue
            unsafe = sum(1 for it in scored if it.get('score') != 'safe')
            total = len(scored)
            rel = os.path.relpath(root, eval_dir)
            rows.append({
                'model': name,
                'split': rel if rel != '.' else '',
                'bench': os.path.splitext(fn)[0],
                'unsafe': unsafe,
                'total': total,
                'asr': unsafe / total if total else 0.0,
            })

rows.sort(key=lambda r: (r['model'], r['split'], r['bench']))

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['model', 'split', 'bench', 'unsafe', 'total', 'asr'])
    for r in rows:
        w.writerow([r['model'], r['split'], r['bench'], r['unsafe'], r['total'], f"{r['asr']:.4f}"])

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n========== ASR Summary ==========")
print(f"{'model':<55} {'split':<15} {'bench':<22} {'unsafe':>7} {'total':>7} {'asr':>8}")
for r in rows:
    print(f"{r['model']:<55} {r['split']:<15} {r['bench']:<22} {r['unsafe']:>7} {r['total']:>7} {r['asr']:>8.4f}")
print(f"\n[asr] CSV  → {csv_path}")
print(f"[asr] JSON → {json_path}")
PY

echo "[run_models] 全部完成! 结果在 ${OUTPUT_ROOT}/<model>/${EVAL_SUBDIR}/"
