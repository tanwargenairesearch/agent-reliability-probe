"""Compile the authoritative reliability dataset from raw checkpoints.

Single source of truth: recomputes every metric from runs/*.jsonl. For each
(domain, model) cell it uses the CLEAN checkpoint — the step-60 rerun where a
step-40 run was contaminated by the step cap, otherwise the original.

Emits results/master_dataset.json (+ a markdown summary on stdout).
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

K = 5

# (label, domain, model, checkpoint, max_steps, clean?, note)
CELLS = [
    ("retail",  "Gemini-3.5-flash", "runs/retail_gemini.jsonl",       40, True,  ""),
    ("retail",  "Kimi-K2.6",        "runs/retail_kimi.jsonl",         40, True,  ""),
    ("airline", "Gemini-3.5-flash", "runs/airline_gemini_s60.jsonl",  60, True,  "rerun @60 (step-40 was capped)"),
    ("airline", "Kimi-K2.6",        "runs/airline_kimi.jsonl",        40, True,  "low cap-rate at 40"),
    ("telecom", "Gemini-3.5-flash", "runs/telecom_gemini_s60.jsonl",  60, "partial", "~95% of failures still hit 60-step cap"),
    ("telecom", "Kimi-K2.6",        "runs/telecom_kimi_s60.jsonl",    60, "partial", "failures still hit 60-step cap"),
]


def load(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if len(r["decisions"]) == K and r["task_id"] not in d:
            d[r["task_id"]] = r["decisions"]
    return d


def pass_hat_k(c, n, k):
    return 0.0 if c < k else comb(c, k) / comb(n, k)


def metrics(d):
    n = len(d)
    pc = [v.count("PASS") for v in d.values()]
    p1 = sum(pc) / (K * n)
    pk = sum(pass_hat_k(c, K, K) for c in pc) / n
    always = sum(1 for c in pc if c == K)
    syst = sum(1 for c in pc if c == 0)
    inter = sum(1 for c in pc if 0 < c < K)
    return {
        "n_tasks": n, "pass@1": round(p1, 3), "pass^5": round(pk, 3),
        "gap": round(p1 - pk, 3),
        "always_pass": always, "intermittent": inter, "systematic": syst,
        "intermittent_share_of_failures": round(inter / (inter + syst), 3) if (inter + syst) else None,
    }


def main():
    rows = []
    for domain, model, ckpt, steps, clean, note in CELLS:
        if not Path(ckpt).exists():
            continue
        m = metrics(load(ckpt))
        m.update(domain=domain, model=model, max_steps=steps, clean=clean,
                 note=note, source=ckpt)
        rows.append(m)

    out = {"k": K, "grading": "tau-bench DB end-state (EvaluationType.ENV)",
           "user_simulator": "gemini/gemini-3.5-flash", "cells": rows}
    Path("results/master_dataset.json").write_text(json.dumps(out, indent=2))

    print("# Master reliability dataset\n")
    print(f"| domain | model | n | pass@1 | pass^5 | gap | always | interm | syst | int% of fails | clean |")
    print(f"|---|---|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    for r in rows:
        cl = "✅" if r["clean"] is True else "⚠️"
        isf = r["intermittent_share_of_failures"]
        print(f"| {r['domain']} | {r['model']} | {r['n_tasks']} | {r['pass@1']:.2f} | "
              f"{r['pass^5']:.2f} | {r['gap']:.2f} | {r['always_pass']} | {r['intermittent']} | "
              f"{r['systematic']} | {f'{isf:.0%}' if isf is not None else '-'} | {cl} |")
    # pooled clean-only intermittent share
    clean = [r for r in rows if r["clean"] is True]
    ti = sum(r["intermittent"] for r in clean); ts = sum(r["systematic"] for r in clean)
    print(f"\nClean cells pooled (retail+airline): {ti} intermittent / {ti+ts} failures = "
          f"{ti/(ti+ts):.0%} intermittent")
    print("\nWrote results/master_dataset.json")


if __name__ == "__main__":
    main()
