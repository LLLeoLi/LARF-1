#!/bin/bash
# ==============================================================================
# run_sorrybench_score.sh
#   入口仍是「模型列表」, 跳过推理, 用 sorry-bench (ft-mistral-7b) 重新打分。
#
#   ★ 新增: 支持跨多个 _models_eval_YYYYMMDD 根目录扫描 ★
#     SF_CKPTS_ROOT 下会有若干 _models_eval_YYYYMMDD/<model>/eval_YYYYMMDD/...
#     设 DATE_FROM / DATE_TO 可框定日期区间, 脚本会自动:
#       - 枚举 SF_CKPTS_ROOT/_models_eval_<YYYYMMDD> 里日期在区间内的目录;
#       - 对每个模型名, 在每个日期目录里都查一次, 命中的都加入打分队列;
#     最终给出一份合并的 ASR 汇总, CSV/JSON 里多一列 eval_date 用来区分。
#
# 输入:
#   bash run_sorrybench_score.sh <model_path_or_id> [<model_path_or_id> ...]
#   MODELS="m1 m2" bash run_sorrybench_score.sh
#   bash run_sorrybench_score.sh -f models.txt
#
# 关键环境变量:
#   多根目录模式 (默认):
#     SF_CKPTS_ROOT   /mnt/hdfs/.../sf_ckpts            根目录, 下含 _models_eval_*
#     DATE_FROM       20260512                          起始日期(含), 留空=不限
#     DATE_TO         20260518                          截止日期(含), 留空=不限
#
#   单根目录模式 (向后兼容, 设了 OUTPUT_ROOT 就走这条):
#     OUTPUT_ROOT     直接指向单个 _models_eval_YYYYMMDD 目录
#
#   其他:
#     EVAL_SUBDIR     eval_YYYYMMDD; 留空 -> 在每个 model 目录下自动取最新 eval_*
#     JUDGE_MODEL     sorry-bench 模型路径
#     JUDGE_PY        sorry_bench_judge.py 路径
#     SCORE_FIELD     写回 JSON 的字段 (默认 score_sorrybench)
#     JUDGE_WORKERS   并发请求数 (默认 32)
#     SUMMARY_OUT_DIR 汇总文件输出目录 (默认 SF_CKPTS_ROOT, 单根模式下默认 OUTPUT_ROOT)
#     INSTRUCTION_KEY / RESPONSE_KEY  强制锁定字段名(默认 instruction / output)
#
# 输出:
#   1) 每个 *.json 新增 score_sorrybench 字段 (原 llama-guard 的 score 字段保留)
#   2) ${SUMMARY_OUT_DIR}/asr_summary_sorrybench_${TAG}.{csv,json}
#      其中 TAG = ${DATE_FROM}_${DATE_TO}_run${RUN_DATE}  (有日期区间时)
#                 或 ${RUN_DATE}                          (单根模式)
#      JSON/CSV 多一列 eval_date 区分不同日期的同名模型。
# ==============================================================================
set +e

ENTRY_DIR="/opt/tiger/entry"
LARF_DIR="${ENTRY_DIR}/LARF"
JUDGE_MODEL="${JUDGE_MODEL:-/mnt/hdfs/tiktok_aiic/user/lihao.612/ft-mistral-7b-instruct-v0.2-sorry-bench-202406}"
JUDGE_PY="${JUDGE_PY:-${LARF_DIR}/sorry_bench_judge.py}"

RUN_DATE=$(date +%Y%m%d)
VLLM_LOG="${LARF_DIR}/_vllm_sorrybench_${RUN_DATE}.log"
SCORE_FIELD="${SCORE_FIELD:-score_sorrybench}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
SERVED_NAME="sorry-bench-judge"
INSTRUCTION_KEY="${INSTRUCTION_KEY:-instruction}"
RESPONSE_KEY="${RESPONSE_KEY:-output}"
# sorry-bench judge 只在 sorry-bench 同分布上训练, 默认只跑 sorry_bench_*.json
# 想跑所有 bench: BENCH_PATTERN="" bash ...
BENCH_PATTERN="${BENCH_PATTERN-sorry_bench}"
# vLLM tokenizer 选择: 
#   auto    -> 默认, 新版 transformers 可能误选 mistral-common (找 tekken.json 找不到)
#   slow    -> 强制 HF python 分词器, 对老 Mistral 格式 (tokenizer.model+tokenizer.json) 最稳
#   mistral -> 强制 mistral-common (新格式, 本模型用不上)
TOKENIZER_MODE="${TOKENIZER_MODE:-slow}"
# 如果模型路径没有 HF tokenizer 文件, 可以单独指向另一份 (e.g. Mistral-7B-Instruct-v0.2)
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

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
    read -r -a MODELS <<< "${MODELS_ENV:-${MODELS}}"
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "[run_sorrybench_score] 用法: bash $0 <model_path> [<model_path> ...]"
    echo "                       或: MODELS=\"m1 m2\" bash $0"
    echo "                       或: bash $0 -f models.txt"
    exit 1
fi

# ------------- 解析根目录: 单根 vs 多根 -------------
OUTPUT_ROOTS=()
if [ -n "${OUTPUT_ROOT:-}" ]; then
    MODE="single"
    OUTPUT_ROOTS=("${OUTPUT_ROOT}")
    SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-${OUTPUT_ROOT}}"
    TAG="${RUN_DATE}"
else
    MODE="multi"
    SF_CKPTS_ROOT="${SF_CKPTS_ROOT:-/mnt/hdfs/tiktok_aiic/user/lihao.612/sf_ckpts}"
    DATE_FROM="${DATE_FROM:-}"
    DATE_TO="${DATE_TO:-}"
    SUMMARY_OUT_DIR="${SUMMARY_OUT_DIR:-${SF_CKPTS_ROOT}}"

    if [ ! -d "${SF_CKPTS_ROOT}" ]; then
        echo "[error] SF_CKPTS_ROOT 不存在: ${SF_CKPTS_ROOT}"
        exit 1
    fi

    # 枚举 _models_eval_YYYYMMDD, 按日期排序, 区间过滤
    for d in "${SF_CKPTS_ROOT}"/_models_eval_*/; do
        d="${d%/}"
        [ -d "$d" ] || continue
        ts=$(basename "$d" | sed 's/^_models_eval_//')
        [[ "$ts" =~ ^[0-9]{8}$ ]] || continue
        if [ -n "$DATE_FROM" ] && [[ "$ts" < "$DATE_FROM" ]]; then continue; fi
        if [ -n "$DATE_TO"   ] && [[ "$ts" > "$DATE_TO"   ]]; then continue; fi
        OUTPUT_ROOTS+=("$d")
    done
    # 按日期升序
    IFS=$'\n' OUTPUT_ROOTS=($(printf "%s\n" "${OUTPUT_ROOTS[@]}" | sort))
    unset IFS

    if [ "${#OUTPUT_ROOTS[@]}" -eq 0 ]; then
        echo "[error] ${SF_CKPTS_ROOT} 下没有符合 [${DATE_FROM:-*}, ${DATE_TO:-*}] 的 _models_eval_YYYYMMDD"
        exit 1
    fi
    TAG="${DATE_FROM:-min}_${DATE_TO:-max}_run${RUN_DATE}"
fi

mkdir -p "${SUMMARY_OUT_DIR}"

echo "[run_sorrybench_score] MODE        = ${MODE}"
echo "[run_sorrybench_score] JUDGE_MODEL = ${JUDGE_MODEL}"
echo "[run_sorrybench_score] JUDGE_PY    = ${JUDGE_PY}"
echo "[run_sorrybench_score] SCORE_FIELD = ${SCORE_FIELD}"
echo "[run_sorrybench_score] BENCH_PAT   = '${BENCH_PATTERN}' (空串=所有 bench)"
echo "[run_sorrybench_score] SUMMARY_OUT = ${SUMMARY_OUT_DIR}"
echo "[run_sorrybench_score] 候选 ROOTS  (${#OUTPUT_ROOTS[@]}):"
for r in "${OUTPUT_ROOTS[@]}"; do echo "    - $r"; done
echo "[run_sorrybench_score] 候选模型    (${#MODELS[@]}):"
for m in "${MODELS[@]}"; do echo "    - $m"; done

# ------------- 在每个 ROOT 下定位每个模型的 eval 目录 -------------
EVAL_BASES=()
PAIR_DATES=()    # 与 EVAL_BASES 等长: 该 base 来自哪个 _models_eval_YYYYMMDD
PAIR_MODELS=()   # 与 EVAL_BASES 等长: 解析后的模型目录名
echo "[run_sorrybench_score] ----- 匹配结果 -----"
for ROOT in "${OUTPUT_ROOTS[@]}"; do
    ROOT_DATE=$(basename "${ROOT}" | sed 's/^_models_eval_//')

    for m in "${MODELS[@]}"; do
        _MP="${m%/}"
        NAME=$(basename "${_MP}")
        if [[ "${NAME}" == checkpoint-* ]]; then
            PARENT=$(basename "$(dirname "${_MP}")")
            NAME="${PARENT}-${NAME}"
        fi
        MODEL_DIR="${ROOT}/${NAME}"
        [ -d "${MODEL_DIR}" ] || continue

        if [ -n "${EVAL_SUBDIR:-}" ]; then
            TARGET="${MODEL_DIR}/${EVAL_SUBDIR}"
        else
            TARGET=$(ls -d "${MODEL_DIR}"/eval_* 2>/dev/null | sort | tail -n 1)
        fi
        if [ -z "${TARGET}" ] || [ ! -d "${TARGET}" ]; then
            echo "  [warn] ${ROOT_DATE} / ${NAME}: 模型目录存在但找不到 eval_*"
            continue
        fi

        EVAL_BASES+=("${TARGET}")
        PAIR_DATES+=("${ROOT_DATE}")
        PAIR_MODELS+=("${NAME}")
        echo "  + [${ROOT_DATE}] ${NAME}"
        echo "      ${TARGET}"
    done
done

if [ "${#EVAL_BASES[@]}" -eq 0 ]; then
    echo "[error] 没有可打分的目录, 退出"
    exit 1
fi
echo "[run_sorrybench_score] 共 ${#EVAL_BASES[@]} 个 (date × model) 待打分"

cd "${LARF_DIR}"

# ============ 阶段 1: 启动 sorry-bench vLLM ============
echo "[run_sorrybench_score] 启动 vLLM (sorry-bench judge) ..."
# 注意:
#   - 用位置参数传 model, --model 在 vllm 0.13 会被移除
#   - --tokenizer-mode slow 强制走 HF 旧式分词器, 绕开 transformers 自动
#     选择 mistral-common 导致的 "No tokenizer file found" 错误
#   - 如果模型自身的 tokenizer 文件缺失/损坏, 可设 TOKENIZER_PATH 指向另一份
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

echo "[run_sorrybench_score] 等待 vLLM (PID=${VLLM_PID}) 就绪 ..."
VLLM_READY=false
for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[run_sorrybench_score] vLLM 就绪 (尝试 ${i} 次)"
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

# ============ 阶段 2: sorry-bench 打分 ============
for i in "${!EVAL_BASES[@]}"; do
    TARGET="${EVAL_BASES[$i]}"
    DT="${PAIR_DATES[$i]}"
    NM="${PAIR_MODELS[$i]}"

    [ -d "${TARGET}" ] || continue
    if ! curl -sf http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[error] vLLM 在打分前已不可达, 日志: ${VLLM_LOG}"
        exit 1
    fi
    mkdir -p "${TARGET}/logs"
    echo ">>> [${DT}] ${NM}  ->  ${TARGET}"
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

# ============ 阶段 3: 汇总 ASR (跨日期合并) ============
SUMMARY_CSV="${SUMMARY_OUT_DIR}/asr_summary_sorrybench_${TAG}.csv"
SUMMARY_JSON="${SUMMARY_OUT_DIR}/asr_summary_sorrybench_${TAG}.json"

# 用 (eval_date :: model_name :: eval_dir) 三元组喂给 python, 避免再去解析路径
TRIPLES=()
for i in "${!EVAL_BASES[@]}"; do
    TRIPLES+=("${PAIR_DATES[$i]}::${PAIR_MODELS[$i]}::${EVAL_BASES[$i]}")
done

python - "${SUMMARY_CSV}" "${SUMMARY_JSON}" "${SCORE_FIELD}" "${BENCH_PATTERN}" "${TRIPLES[@]}" <<'PY'
import csv, json, os, sys

csv_path, json_path, score_field, bench_pattern, *triples = sys.argv[1:]

rows = []
for t in triples:
    eval_date, model_name, eval_dir = t.split("::", 2)
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
                "eval_date": eval_date,
                "model": model_name,
                "split": rel if rel != "." else "",
                "bench": os.path.splitext(fn)[0],
                "unsafe": unsafe,
                "total": total,
                "asr": unsafe / total if total else 0.0,
            })

rows.sort(key=lambda r: (r["eval_date"], r["model"], r["split"], r["bench"]))

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["eval_date", "model", "split", "bench", "unsafe", "total", "asr"])
    for r in rows:
        w.writerow([r["eval_date"], r["model"], r["split"], r["bench"],
                    r["unsafe"], r["total"], f"{r['asr']:.4f}"])

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print("\n========== ASR Summary (sorry-bench judge) ==========")
print(f"{'eval_date':<10} {'model':<55} {'split':<10} {'bench':<22} {'unsafe':>7} {'total':>7} {'asr':>8}")
for r in rows:
    print(f"{r['eval_date']:<10} {r['model']:<55} {r['split']:<10} {r['bench']:<22} "
          f"{r['unsafe']:>7} {r['total']:>7} {r['asr']:>8.4f}")
print(f"\n[asr] CSV  → {csv_path}")
print(f"[asr] JSON → {json_path}")
PY

echo "[run_sorrybench_score] 全部完成!"
echo "  Per-sample 字段:   ${SCORE_FIELD}"
echo "  汇总 JSON:         ${SUMMARY_JSON}"
echo "  汇总 CSV:          ${SUMMARY_CSV}"