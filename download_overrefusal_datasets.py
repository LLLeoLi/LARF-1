#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
download_overrefusal_datasets.py
================================
把 over-refusal 评测集 (XSTest v2 + OR-Bench) 下载到本地 datasets/ 目录,
转成与 eval_student_model_usedatasets.py 兼容的 CSV (必须有 `Goal` 列)。

生成的文件 (默认写到 ./datasets):
  xstest_safe.csv        250 条「看着危险其实安全」的 prompt  —— 过度拒绝 = 坏
  xstest_unsafe.csv      200 条真正不安全的 contrast prompt    —— 拒绝       = 好
  or_bench_hard_1k.csv   1319 条最难的安全 prompt              —— 过度拒绝 = 坏
  or_bench_toxic.csv     655 条真正有害的 prompt               —— 拒绝       = 好

数据来源:
  XSTest : https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv
           (列: id,prompt,type,label,focus,note; label ∈ {safe,unsafe})
  OR-Bench: HuggingFace `bench-llm/or-bench`, 通过 dataset-viewer rows API 分页抓取
           (本机没装 pyarrow, 所以不直接读 parquet, 改走 JSON rows API)

只用到标准库 + requests, 不依赖 datasets / pyarrow。
"""

import os
import csv
import sys
import time
import json
import argparse
import urllib.request
import urllib.parse

XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
OR_BENCH_DATASET = "bench-llm/or-bench"


def _http_get(url, timeout=60, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"GET 失败 ({retries} 次): {url}\n  {last}")


def _write_csv(path, rows, extra_cols):
    """rows: list[dict]; 每行至少有 'Goal'; extra_cols 是额外保留的列名顺序。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["Goal"] + [c for c in extra_cols if c != "Goal"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  [写出] {path}  ({len(rows)} 条)")


# ----------------------------------------------------------------------------
# XSTest
# ----------------------------------------------------------------------------
def download_xstest(out_dir):
    print("[XSTest] 下载 ...", XSTEST_URL)
    raw = _http_get(XSTEST_URL).decode("utf-8")
    reader = csv.DictReader(raw.splitlines())
    safe, unsafe = [], []
    for row in reader:
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        rec = {
            "Goal": prompt,
            "type": row.get("type", ""),
            "label": row.get("label", ""),
            "focus": row.get("focus", ""),
            "note": row.get("note", ""),
        }
        if (row.get("label") or "").strip().lower() == "safe":
            safe.append(rec)
        else:
            unsafe.append(rec)
    cols = ["type", "label", "focus", "note"]
    _write_csv(os.path.join(out_dir, "xstest_safe.csv"), safe, cols)
    _write_csv(os.path.join(out_dir, "xstest_unsafe.csv"), unsafe, cols)


# ----------------------------------------------------------------------------
# OR-Bench (HF dataset-viewer rows API, 100 行/页)
# ----------------------------------------------------------------------------
def _fetch_orbench_config(config):
    rows = []
    offset = 0
    page = 100
    total = None
    while True:
        q = urllib.parse.urlencode({
            "dataset": OR_BENCH_DATASET,
            "config": config,
            "split": "train",
            "offset": offset,
            "length": page,
        })
        data = json.loads(_http_get(f"{HF_ROWS_API}?{q}").decode("utf-8"))
        if total is None:
            total = data.get("num_rows_total")
        batch = data.get("rows", [])
        if not batch:
            break
        for item in batch:
            r = item.get("row", {})
            prompt = (r.get("prompt") or "").strip()
            if prompt:
                rows.append({"Goal": prompt, "category": r.get("category", "")})
        offset += page
        if total is not None and offset >= total:
            break
    return rows


def download_orbench(out_dir, include_80k=False):
    plan = [
        ("or-bench-hard-1k", "or_bench_hard_1k.csv"),
        ("or-bench-toxic", "or_bench_toxic.csv"),
    ]
    if include_80k:
        plan.append(("or-bench-80k", "or_bench_80k.csv"))
    for config, fname in plan:
        print(f"[OR-Bench] 抓取 {config} ...")
        rows = _fetch_orbench_config(config)
        _write_csv(os.path.join(out_dir, fname), rows, ["category"])


def main():
    ap = argparse.ArgumentParser(description="下载 XSTest + OR-Bench 到本地 CSV")
    ap.add_argument("--out_dir", default="datasets", help="输出目录 (默认 ./datasets)")
    ap.add_argument("--only", choices=["xstest", "orbench", "all"], default="all")
    ap.add_argument("--include_80k", action="store_true",
                    help="额外下载 or-bench-80k (80k 条, 很大, 默认不下)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.only in ("xstest", "all"):
        download_xstest(args.out_dir)
    if args.only in ("orbench", "all"):
        download_orbench(args.out_dir, include_80k=args.include_80k)
    print("[done] 全部下载完成 ->", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
