"""Extract diagnostic features per intermittent task to support failure labeling.

For each task with both PASS and FAIL transcripts, compares the runs and surfaces
signals that map to failure categories:
  - missing_action : a write/action tool present in PASS runs but absent in FAIL
  - arg_divergence : same tool multiset, so failure is in arguments/values
  - shorter_fail   : FAIL runs end with fewer messages (gave up early)
  - bad_calculate  : calculate() args look malformed in FAIL

Output is a per-task feature row; final category is assigned by reading, but
these features make the labeling fast and consistent.
"""

from __future__ import annotations

import glob
import json
from collections import Counter

WRITE_TOOLS = {
    "modify_pending_order_items", "exchange_delivered_order_items",
    "return_delivered_order_items", "cancel_pending_order",
    "modify_pending_order_address", "modify_user_address",
}


def _load(f):
    r = json.load(open(f))
    tools = []
    nmsg = len(r["messages"])
    calcs = []
    for m in r["messages"]:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name") or tc.get("name", "?")
            tools.append(name)
            if name == "calculate":
                calcs.append(str(fn.get("arguments") or tc.get("arguments", "")))
    return r["decision"], nmsg, tools, calcs


def task_features(task_dir, task_id):
    files = sorted(glob.glob(f"{task_dir}/retail_{task_id}_trial*.json"))
    if not files:
        return None
    runs = [_load(f) for f in files]
    passes = [r for r in runs if r[0] == "PASS"]
    fails = [r for r in runs if r[0] == "FAIL"]
    if not passes or not fails:
        return {"task": task_id, "kind": "systematic" if not passes else "all_pass",
                "n_pass": len(passes), "n_fail": len(fails)}
    pass_tools = Counter(passes[0][2])
    fail_tools = Counter(fails[0][2])
    pass_writes = {t for t in pass_tools if t in WRITE_TOOLS}
    fail_writes = {t for t in fail_tools if t in WRITE_TOOLS}
    missing_action = sorted(pass_writes - fail_writes)
    same_toolset = (set(pass_tools) == set(fail_tools))
    avg_pass_msgs = sum(r[1] for r in passes) / len(passes)
    avg_fail_msgs = sum(r[1] for r in fails) / len(fails)
    return {
        "task": task_id,
        "n_pass": len(passes), "n_fail": len(fails),
        "missing_action_in_fail": missing_action,         # -> "stalled / no action"
        "same_toolset": same_toolset,                     # -> "wrong arguments"
        "fail_shorter_by": round(avg_pass_msgs - avg_fail_msgs, 1),
        "pass_tools": dict(pass_tools),
        "fail_tools": dict(fail_tools),
    }


def main():
    import sys
    task_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/transcripts_tax"
    ids = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    if not ids:
        # infer from files
        ids = sorted({f.split("_")[1] for f in glob.glob(f"{task_dir}/retail_*_trial*.json")}, key=int)
    rows = [task_features(task_dir, t) for t in ids]
    rows = [r for r in rows if r]
    for r in rows:
        if r.get("kind"):
            print(f"task {r['task']:>3}: {r['kind']} (pass={r['n_pass']} fail={r['n_fail']})")
            continue
        sig = []
        if r["missing_action_in_fail"]:
            sig.append(f"STALLED→no {','.join(r['missing_action_in_fail'])}")
        elif r["same_toolset"]:
            sig.append("WRONG-ARGS (same tools)")
        else:
            sig.append("DIFFERENT-PATH")
        if r["fail_shorter_by"] >= 3:
            sig.append(f"fail {r['fail_shorter_by']} msgs shorter")
        print(f"task {r['task']:>3}: {' | '.join(sig)}  (pass={r['n_pass']} fail={r['n_fail']})")


if __name__ == "__main__":
    main()
