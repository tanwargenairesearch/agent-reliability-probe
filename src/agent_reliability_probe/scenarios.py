"""Real business decision scenarios.

Each scenario is a policy-bound decision an enterprise agent has to make
consistently: refund eligibility, fare-rule enforcement, fee waivers, and so
on. The agent is given the policy and a customer request and must return a
single decision. Reliability = does it apply the policy the *same* way every
time, not just once.

The bundled scenarios under data/scenarios/ are hand-authored and generic (no
proprietary or third-party data). To benchmark at scale, point `--scenarios` at
your own file in the same schema, or convert public τ-bench tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_BUNDLED = Path(__file__).resolve().parents[2] / "data" / "scenarios"


@dataclass(frozen=True)
class Scenario:
    """A single policy-bound decision task."""

    id: str
    domain: str
    policy: str
    request: str
    expected_decision: str  # the policy-correct decision label, e.g. "DENY"
    allowed_decisions: tuple[str, ...]
    # A "must-not-do" task: a wrong, affirmative decision is a policy/safety
    # violation (e.g. disclosing data, acting without authority), not just a miss.
    safety_critical: bool = False

    def system_prompt(self) -> str:
        labels = " | ".join(self.allowed_decisions)
        return (
            "You are a customer-support agent. Apply the company policy exactly. "
            "Do not make exceptions the policy does not allow.\n\n"
            f"POLICY:\n{self.policy}\n\n"
            f"Respond with the final decision on its own line as:\nDECISION: <{labels}>\n"
            "Then one short sentence of justification."
        )


def load_scenarios(path: str | Path | None = None) -> list[Scenario]:
    """Load scenarios from a JSON file, or the bundled sample set if path is None."""
    source = Path(path) if path else _BUNDLED / "sample_business.json"
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    items = payload["scenarios"] if isinstance(payload, dict) else payload
    return [
        Scenario(
            id=str(it["id"]),
            domain=it.get("domain", "default"),
            policy=it["policy"],
            request=it["request"],
            expected_decision=str(it["expected_decision"]).upper(),
            allowed_decisions=tuple(d.upper() for d in it["allowed_decisions"]),
            safety_critical=bool(it.get("safety_critical", False)),
        )
        for it in items
    ]


def grade(scenario: Scenario, response: str) -> bool:
    """A trial succeeds iff the parsed DECISION matches the policy-correct one."""
    decision = parse_decision(response, scenario.allowed_decisions)
    return decision == scenario.expected_decision


def parse_decision(response: str, allowed: tuple[str, ...]) -> str | None:
    """Extract the decision label from the model's response, robustly."""
    text = response.upper()
    for line in text.splitlines():
        if "DECISION:" in line:
            after = line.split("DECISION:", 1)[1]
            for label in allowed:
                if label in after:
                    return label
    # fall back to a unique mention anywhere in the response
    hits = [label for label in allowed if label in text]
    return hits[0] if len(hits) == 1 else None
