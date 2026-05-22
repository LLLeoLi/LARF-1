#!/bin/bash
# ==============================================================================
# run_teacher_ablation_sorry.sh
#   为 run_teacher_ablation.sh 生成的 JSON 用 sorry-bench (ft-mistral-7b) 重新打分.
#
#   输出目录结构 (来自 run_teacher_ablation.sh):
#     ${OUTPUT_ROOT}/<model>__<mode>/eval_<DATE>/{temp_0,temp_1,temp_1_run2,temp_1_run3}/*.json
#
#   用法 (只接受 models.txt):
#     bash run_teacher_ablation_sorry.sh -f models.txt
#
#   关键环境变量:
#     OUTPUT_ROOT     默认与 run_teacher_ablation.sh 一致: <repo>/_ablation_eval_<RUN_DATE>
#     MODES           默认 "system_prompt steering"
#     EVAL_SUBDIR     默认在每个 <model>__<mode>/ 下自动取最新 eval_*
#     JUDGE_MODEL     sorry-bench 模型路径
#     JUDGE_PY        sorry_bench_judge.py 路径
#     SCORE_FIELD     写回字段 (默认 score_sorrybench)
#     JUDGE_WORKERS   并发请求数 (默认 32)
#     BENCH_PATTERN   只处理文件名含该子串的 *.json (默认 sorry_bench; 设 "" = 所有 bench)
#     INSTRUCTION_KEY / RESPONSE_KEY  默认 instruction / output
#     TOKENIZER_MODE  默认 slow (绕开 mistral-common)
#     TOKENIZER_PATH  备用 tokenizer 路径
#
#   输出:
#     ${OUTPUT_ROOT}/asr_summary_sorrybench_${RUN_DATE}.{csv,json}
#     CSV/JSON 含 model / mode / split / bench / unsafe / total / asr 列
# ==============================================================================
set +e

LARF_DIR="${LARF_DIR:-$(cd "$(dirname "$0")" && pwd)}"
REPO_DIR="$(dirname "${LARF_DIR}")"

ENTRY_DIR="/opt/tiger/entry"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/hdfs/tiktok_aiic/user/lihao.612/ft-mistral-7b-instruct-v0.2-sorry-bench-202406}"
JUDGE_PY="${JUDGE_PY:-${LARF_DIR}/sorry_bench_judge.py}"

RUN_DATE=$(date +%Y%m%d)
VLLM_LOG="${LARF_DIR}/_vllm_ablation_sorry_${RUN_DATE}.log"
SCORE_FIELD="${SCORE_FIELD:-score_sorrybench}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
SERVED_NAME="sorry-bench-judge"
INSTRUCTION_KEY="${INSTRUCTION_KEY:-instruction}"
RESPONSE_KEY="${RESPONSE_KEY:-output}"
BENCH_PATTERN="${BENCH_PATTERN-sorry_bench}"
TOKENIZER_MODE="${TOKENIZER_MODE:-slow}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

# 默认输出根目录跟 run_teacher_ablation.sh 对齐
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/_ablation_eval_${RUN_DATE}}"
SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-${OUTPUT_ROOT}}"

MODES_DEFAULT="system_prompt steering"
read -r -a MODES <<< "${MODES:-${MODES_DEFAULT}}"

# ------------- 解析模型列表 (仅 -f models.txt) -------------
MODELS=()
if [ "$1" = "-f" ] && [ -n "$2" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line//$'\r'/}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -n "$line" ] && MODELS+=("$line")
    done < "$2"
else
    echo "[run_teacher_ablation_sorry] 用法: bash $0 -f models.txt"
    exit 1
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[error] models.txt 为空"
    exit 1
fi

if [ ! -d "${OUTPUT_ROOT}" ]; then
    echo "[error] OUTPUT_ROOT 不存在: ${OUTPUT_ROOT}"
    exit 1
fi

mkdir -p "${SUMMARY_OUT_DIR}"

echo "[run_teacher_ablation_sorry] OUTPUT_ROOT = ${OUTPUT_ROOT}"
echo "[run_teacher_ablation_sorry] MODES       = ${MODES[*]}"
echo "[run_teacher_ablation_sorry] JUDGE_MODEL = ${JUDGE_MODEL}"
echo "[run_teacher_ablation_sorry] SCORE_FIELD = ${SCORE_FIELD}"
echo "[run_teacher_ablation_sorry] BENCH_PAT   = '${BENCH_PATTERN}' (空串=所有 bench)"
echo "[run_teacher_ablation_sorry] 候选模型 (${#MODELS[@]}):"
for m in "${MODELS[@]}"; do echo "    - $m"; done

# ------------- 定位 (model, mode) -> eval_* 目录 -------------
EVAL_BASES=()
PAIR_MODELS=()   # 与 EVAL_BASES 等长
PAIR_MODES=()
echo "[run_teacher_ablation_sorry] ----- 匹配结果 -----"
for m in "${MODELS[@]}"; do
    _MP="${m%/}"
    NAME=$(basename "${_MP}")
    if [[ "${NAME}" == checkpoint-* ]]; then
        PARENT=$(basename "$(dirname "${_MP}")")
        NAME="${PARENT}-${NAME}"
    fi

    for MODE in "${MODES[@]}"; do
        TAG="${NAME}__${MODE}"
        MODEL_DIR="${OUTPUT_ROOT}/${TAG}"
        if [ ! -d "${MODEL_DIR}" ]; then
            echo "  [warn] 缺失目录: ${MODEL_DIR}"
            continue
        fi
        if [ -n "${EVAL_SUBDIR:-}" ]; then
            TARGET="${MODEL_DIR}/${EVAL_SUBDIR}"
        else
            TARGET=$(ls -d "${MODEL_DIR}"/eval_* 2>/dev/null | sort | tail -n 1)
        fi
        if [ -z "${TARGET}" ] || [ ! -d "${TARGET}" ]; then
            echo "  [warn] ${TAG}: 找不到 eval_*"
            continue
        fi
        EVAL_BASES+=("${TARGET}")
        PAIR_MODELS+=("${NAME}")
        PAIR_MODES+=("${MODE}")
        echo "  + ${TAG}"
        echo "      ${TARGET}"
    done
done

if [ "${#EVAL_BASES[@]}" -eq 0 ]; then
    echo "[error] 没有可打分的目录"
    exit 1
fi
echo "[run_teacher_ablation_sorry] 共 ${#EVAL_BASES[@]} 个 (model × mode) 待打分"

cd "${LARF_DIR}"

# ============ 阶段 1: 启动 sorry-bench vLLM ============
echo "[run_teacher_ablation_sorry] 启动 vLLM (sorry-bench judge) ..."
VLLM_EXTRA_ARGS=()
[ -n "${TOKENIZER_PATH}" ] && VLLM_EXTRA_ARGS+=(--tokenizer "${TOKENIZER_PATH}")
vllm serve "${JUDGE_MODEL}" \
    --port=8000 \
    --max-model-len 8192 \
    --data-parallel-size=4 \
    --trust-remote-code \
    --tokenizer-mode "${TOKENIZER_MODE}" \
    --served-model-name "${SERVED_NAME}" \
    "${VLLM_EXTRA_ARGS[@]}" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

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

echo "[run_teacher_ablation_sorry] 等待 vLLM (PID=${VLLM_PID}) 就绪 ..."
VLLM_READY=false
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[run_teacher_ablation_sorry] vLLM 就绪 (尝试 ${i} 次)"
        VLLM_READY=true; break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[error] vLLM 进程已退出, 日志: ${VLLM_LOG}"
        exit 1
    fi
    sleep 5
done
[ "${VLLM_READY}" = "true" ] || { echo "[error] vLLM 未就绪"; exit 1; }

# ============ 阶段 2: sorry-bench 打分 ============
for i in "${!EVAL_BASES[@]}"; do
    TARGET="${EVAL_BASES[$i]}"
    NM="${PAIR_MODELS[$i]}"
    MD="${PAIR_MODES[$i]}"

    [ -d "${TARGET}" ] || continue
    if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[error] vLLM 在打分前已不可达"
        exit 1
    fi
    mkdir -p "${TARGET}/logs"
    echo ">>> [${MD}] ${NM}  ->  ${TARGET}"
    python "${JUDGE_PY}" \
        --input_dir       "${TARGET}" \
        --judge_model     "${SERVED_NAME}" \
        --client_url      http://localhost:8000 \
        --score_field     "${SCORE_FIELD}" \
        --max_workers     "${JUDGE_WORKERS}" \
        --instruction_key "${INSTRUCTION_KEY}" \
        --response_key    "${RESPONSE_KEY}" \
        --bench_pattern   "${BENCH_PATTERN}" \
        2>&1 | tee "${TARGET}/logs/sorrybench.log"
done

# ============ 阶段 3: 汇总 ASR ============
SUMMARY_CSV="${SUMMARY_OUT_DIR}/asr_summary_sorrybench_${RUN_DATE}.csv"
SUMMARY_JSON="${SUMMARY_OUT_DIR}/asr_summary_sorrybench_${RUN_DATE}.json"

TRIPLES=()
for i in "${!EVAL_BASES[@]}"; do
    TRIPLES+=("${PAIR_MODELS[$i]}::${PAIR_MODES[$i]}::${EVAL_BASES[$i]}")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${SCORE_FIELD}" "${BENCH_PATTERN}" "${TRIPLES[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, score_field, bench_pattern, *triples = sys.argv[1:]

rows = []
for t in triples:
    model_name, mode, eval_dir = t.split("::", 2)
    if not os.path.isdir(eval_dir):
        continue
    for root, _, files in os.walk(eval_dir):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            if bench_pattern and bench_pattern not in fn:
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[asr] WARN: 读取失败 {fp}: {e}")
                continue
            if not isinstance(data, list) or not data:
                continue
            scored = [
                it for it in data
                if isinstance(it, dict)
                and score_field in it
                and it[score_field] in (0, 1)
            ]
            if not scored:
                continue
            # sorry-bench: 1 = 合规于有害请求 (unsafe), 0 = 拒绝 (safe)
            unsafe = sum(1 for it in scored if int(it[score_field]) == 1)
            total = len(scored)
            rel = os.path.relpath(root, eval_dir)
            rows.append({
                "model": model_name,
                "mode": mode,
                "split": rel if rel != "." else "",
                "bench": os.path.splitext(fn)[0],
                "unsafe": unsafe,
                "total": total,
                "asr": unsafe / total if total else 0.0,
            })

rows.sort(key=lambda r: (r["model"], r["mode"], r["split"], r["bench"]))

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "mode", "split", "bench", "unsafe", "total", "asr"])
    for r in rows:
        w.writerow([r["model"], r["mode"], r["split"], r["bench"],
                    r["unsafe"], r["total"], f"{r['asr']:.4f}"])

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n========== ASR Summary (sorry-bench judge, ablation) ==========")
print(f"{'model':<45} {'mode':<14} {'split':<12} {'bench':<22} {'unsafe':>7} {'total':>7} {'asr':>8}")
for r in rows:
    print(f"{r['model']:<45} {r['mode']:<14} {r['split']:<12} {r['bench']:<22} "
          f"{r['unsafe']:>7} {r['total']:>7} {r['asr']:>8.4f}")
print(f"\n[asr] CSV  -> {csv_path}")
print(f"[asr] JSON -> {json_path}")
PY

echo "[run_teacher_ablation_sorry] 全部完成!"
echo "  Per-sample 字段: ${SCORE_FIELD}"
echo "  汇总 CSV : ${SUMMARY_CSV}"
echo "  汇总 JSON: ${SUMMARY_JSON}"
