"""Classify every FAILING trajectory by its own evidence (no PASS-pair needed).

Each failing run is bucketed into a failure mode from observable signals:

  no_action      : the task needs a write/action tool, but none was ever called
                   -> "Stalled — gathered info, never executed the action"
  agent_quit     : the AGENT ended the conversation (transfer-to-human / give-up)
                   while the task was unfinished
  loop_or_cap    : hit the step cap / very long with repeated tool calls
                   -> "Lost the thread / looped"
  acted_wrong    : a write tool WAS called but the end-state was wrong
                   -> "Acted with wrong arguments/values"

Heuristic but transparent; prints per-trajectory label + the aggregate taxonomy.
Run:  python results/classify_failures.py runs/transcripts_tax runs/transcripts_pilot
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import Counter

WRITE_TOOLS = {
    # retail
    "modify_pending_order_items", "exchange_delivered_order_items",
    "return_delivered_order_items", "cancel_pending_order",
    "modify_pending_order_address", "modify_user_address",
    # airline
    "update_reservation_flights", "update_reservation_passengers",
    "update_reservation_baggages", "cancel_reservation", "book_reservation",
    "send_certificate",
    # telecom (state-changing line/device actions)
    "enable_roaming", "disable_roaming", "refuel_data", "send_payment_request",
    "suspend_line", "resume_line", "reactivate_line",
}
# escalating to a human while the task is unfinished is its own mode, not a stall
ESCALATE = {"transfer_to_human_agents"}


def classify(path):
    r = json.load(open(path))
    if r.get("decision") != "FAIL":
        return None
    msgs = r["messages"]
    nmsg = len(msgs)
    term = str(r.get("termination_reason", ""))
    tools, write_calls, escalated = [], 0, False
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            name = fn.get("name") or tc.get("name", "?")
            tools.append(name)
            if name in WRITE_TOOLS:
                write_calls += 1
            if name in ESCALATE:
                escalated = True
    # precedence: ran-out-of-budget > escalated > acted-wrong > stalled
    if "MAX_STEP" in term.upper():
        label = "Lost the thread / looped (hit step cap)"
    elif escalated and write_calls == 0:
        label = "Escalated to human without completing"
    elif write_calls > 0:
        label = "Acted with wrong arguments/values"
    else:
        label = "Stalled — never executed the action"
    return {"file": os.path.basename(path), "label": label,
            "nmsg": nmsg, "writes": write_calls, "term": term.split(".")[-1]}


def main():
    dirs = sys.argv[1:] or ["runs/transcripts_tax"]
    files = []
    for d in dirs:
        files += glob.glob(f"{d}/*.json")
    # dedup by basename (pilot+tax overlap on tasks 20,21,28,29,30)
    seen, uniq = set(), []
    for f in sorted(files):
        b = os.path.basename(f)
        if b not in seen:
            seen.add(b); uniq.append(f)
    rows = [classify(f) for f in uniq]
    rows = [r for r in rows if r]
    cats = Counter(r["label"] for r in rows)
    tot = sum(cats.values())
    print(f"=== FAILURE TAXONOMY — all {tot} failing trajectories ===\n")
    for c, n in cats.most_common():
        print(f"  {n:3d}  ({100*n/tot:3.0f}%)  {c}")
    print(f"\n  total failing trajectories classified: {tot}")
    # show a few examples per category
    print("\n--- examples ---")
    for c, _ in cats.most_common():
        ex = [r for r in rows if r["label"] == c][:3]
        print(f"  {c}:")
        for r in ex:
            print(f"      {r['file']}  msgs={r['nmsg']} writes={r['writes']} end={r['term']}")


if __name__ == "__main__":
    main()
