#!/bin/bash
# ==============================================================================
# run.sh — LARF 安全能力测试
#   阶段 1: 并行跑 eval_student_model_usedatasets.py (temp=0 一次 + temp=1 三次)
#   阶段 2: vLLM 起 Llama-Guard 服务 (port 8000, GPU 0)
#   阶段 3: llama_guard.py 给 JSON 打分
# ==============================================================================
set +e

ENTRY_DIR="/opt/tiger/entry"
LARF_DIR="${ENTRY_DIR}/LARF"
LLAMA_GUARD_MODEL="${ENTRY_DIR}/Llama-Guard-4-12B"

# 训练脚本把产物放到 sf_ckpts/${MODEL_BASE}/${MODEL_BASE}-vector-...
# 切换被测 family: CKPT_BASE=Qwen3-8B bash run.sh
CKPT_BASE="${CKPT_BASE:-Llama-3-8B-Instruct}"
CKPT_ROOT="/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts/${CKPT_BASE}-0505-new"
CKPT_GLOB="${CKPT_GLOB:-${CKPT_BASE}-vector-*-PKU_UnSafeRLHF_100}"

DATE_TAG=$(date +%Y%m%d)
EVAL_SUBDIR="eval_${DATE_TAG}"
# vLLM 日志走本地 (HDFS fuse 写日志看不到实时输出, 不利于调试)
VLLM_LOG="${LARF_DIR}/_vllm_${DATE_TAG}.log"

BATCH_SIZE=8
# 与 run_safe_system_prompt_new.sh 一致: LARF/datasets/ 下 7 个 csv (Goal 列)
DATA_DIR="datasets"
BENCHES=(advbench ALERT HarmfulQA JBB-Behaviors PKU-SafeRLHF-30K sorry_bench_202503 harmbench)

shopt -s nullglob
MODELS=("${CKPT_ROOT}"/${CKPT_GLOB})
shopt -u nullglob
echo "[run] 发现 ${#MODELS[@]} 个模型, 输出 → 各模型目录下 ${EVAL_SUBDIR}/"

cd "${LARF_DIR}"

# ============ 阶段 1: 推理 ============
for i in "${!MODELS[@]}"; do
    MODEL_PATH="${MODELS[$i]}"
    GPU_ID=$(( i % BATCH_SIZE ))
    NAME=$(basename "${MODEL_PATH}")

    EVAL_PATH=""; LATEST=-1
    shopt -s nullglob
    for c in "${MODEL_PATH}"/checkpoint-*; do
        [ -d "$c" ] || continue
        step="${c##*/checkpoint-}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step > LATEST )) && LATEST=$step && EVAL_PATH="$c"
    done
    shopt -u nullglob
    [ -z "${EVAL_PATH}" ] && { echo "[run] WARN: ${NAME} 无 checkpoint, 跳过"; continue; }

    OUT_BASE="${MODEL_PATH}/${EVAL_SUBDIR}"
    mkdir -p "${OUT_BASE}/logs"
    LOG="${OUT_BASE}/logs/eval.log"
    echo ">>> [GPU ${GPU_ID}] ${NAME} → $(basename ${EVAL_PATH})"
    (
        CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model_usedatasets.py \
            --model_path "${EVAL_PATH}" --output_dir "${OUT_BASE}/temp_0" \
            --data_dir "${DATA_DIR}" \
            --temperature 0 --benches "${BENCHES[@]}"
        for r in 1 2 3; do
            sub="temp_1"; [ "$r" -gt 1 ] && sub="temp_1_run${r}"
            CUDA_VISIBLE_DEVICES=${GPU_ID} python eval_student_model_usedatasets.py \
                --model_path "${EVAL_PATH}" --output_dir "${OUT_BASE}/${sub}" \
                --data_dir "${DATA_DIR}" \
                --temperature 1 --benches "${BENCHES[@]}"
        done
    ) > "${LOG}" 2>&1 &

    (( (i + 1) % BATCH_SIZE == 0 )) && wait
done
wait
echo "[run] 阶段 1 完成"

# ============ 阶段 2: vLLM Llama-Guard 服务 (GPU 0, port 8000) ============
# 先启动后台进程
vllm serve --model "${LLAMA_GUARD_MODEL}" --port=8000  --max-model-len 1000 \
    --data-parallel-size=8 --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

# 确保 PID 有效后再注册清理
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

echo "[run] 等待 vLLM (PID=${VLLM_PID}) 就绪 ..."
VLLM_READY=false
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[run] vLLM 就绪 (尝试 ${i} 次)"
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
for MODEL_PATH in "${MODELS[@]}"; do
    NAME=$(basename "${MODEL_PATH}")
    TARGET="${MODEL_PATH}/${EVAL_SUBDIR}"
    [ -d "${TARGET}" ] || continue

    # 打分前检查服务存活
    if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[error] vLLM 服务在打分前已不可达, 日志: ${VLLM_LOG}"
        exit 1
    fi

    mkdir -p "${TARGET}/logs"
    echo ">>> 打分: ${TARGET}"
    python "${LARF_DIR}/llama_guard.py" --input_dir "${TARGET}" 2>&1 | tee "${TARGET}/logs/guard.log"
done

# ============ 阶段 4: 汇总 ASR ============
SUMMARY_CSV="${CKPT_ROOT}/asr_summary_${DATE_TAG}.csv"
SUMMARY_JSON="${CKPT_ROOT}/asr_summary_${DATE_TAG}.json"

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${EVAL_SUBDIR}" "${MODELS[@]}" <<'PY'
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

echo "[run] 全部完成! 结果在各模型目录下 ${EVAL_SUBDIR}/"
