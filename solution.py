import json
import re
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryRule:
    name: str
    positive_groups: List[List[str]]   # OR within group, AND across groups
    negative_signals: List[str] = field(default_factory=list)
    confidence_boosters: List[str] = field(default_factory=list)
    base_confidence: float = 0.7

    def matches(self, text: str) -> Tuple[bool, float]:
        text_l = text.lower()
        # Check negatives first
        for neg in self.negative_signals:
            if neg.lower() in text_l:
                return False, 0.0
        # Each positive group must have at least one hit
        group_hits = 0
        for group in self.positive_groups:
            if any(kw.lower() in text_l for kw in group):
                group_hits += 1
        if group_hits < len(self.positive_groups):
            return False, 0.0
        # Confidence boosting
        confidence = self.base_confidence
        for booster in self.confidence_boosters:
            if booster.lower() in text_l:
                confidence = min(confidence + 0.1, 1.0)
        return True, round(confidence, 2)


# ---------------------------------------------------------------------------
# Query rules definition
# ---------------------------------------------------------------------------

QUERY_RULES: Dict[str, QueryRule] = {
    "ev_battery_supply_chain": QueryRule(
        name="ev_battery_supply_chain",
        positive_groups=[
            ["battery", "lithium", "cathode", "anode", "electrolyte", "cell manufacturing"],
            ["electric vehicle", "ev ", "bev", "phev", "e-mobility", "traction"],
        ],
        negative_signals=["battery recycling only", "lead acid", "alkaline battery"],
        confidence_boosters=["gigafactory", "battery management system", "bms", "solid state", "ncm", "nca", "lfp"],
        base_confidence=0.75,
    ),
    "fintech_growth_proxy": QueryRule(
        name="fintech_growth_proxy",
        positive_groups=[
            ["payment", "lending", "neobank", "digital bank", "insurtech", "wealthtech", "regtech", "embedded finance"],
            ["series a", "series b", "series c", "seed round", "venture", "raised", "funding", "growth"],
        ],
        negative_signals=["traditional bank", "brick and mortar", "legacy banking"],
        confidence_boosters=["api-first", "open banking", "bnpl", "crypto", "defi", "fintech"],
        base_confidence=0.70,
    ),
    "supplier_role_detection": QueryRule(
        name="supplier_role_detection",
        positive_groups=[
            ["supplier", "manufacturer", "oem", "tier 1", "tier 2", "component", "parts", "fabricat"],
            ["automotive", "aerospace", "industrial", "machinery", "electronics", "semiconductor"],
        ],
        negative_signals=["retailer", "distributor only", "end consumer"],
        confidence_boosters=["iso 9001", "iatf", "as9100", "supply chain", "procurement"],
        base_confidence=0.68,
    ),
}


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def build_text(company: dict) -> str:
    fields = [
        company.get("company_name", ""),
        company.get("long_description", ""),
        company.get("short_description", ""),
        company.get("business_tags", ""),
        company.get("sector", ""),
        company.get("category", ""),
        company.get("nace_code_label", ""),
        company.get("specialties", ""),
        company.get("keywords", ""),
    ]
    parts = []
    for f in fields:
        if isinstance(f, list):
            parts.append(" ".join(str(x) for x in f))
        elif f:
            parts.append(str(f))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Reranker for overlapping matches
# ---------------------------------------------------------------------------

def rerank(matches: List[Tuple[str, float]], text: str) -> List[Tuple[str, float]]:
    """Apply lightweight reranking to resolve ambiguous overlaps."""
    if len(matches) <= 1:
        return matches
    text_l = text.lower()
    scored = []
    for name, conf in matches:
        bonus = 0.0
        if name == "ev_battery_supply_chain" and ("electric vehicle" in text_l or "ev battery" in text_l):
            bonus += 0.05
        if name == "fintech_growth_proxy" and "fintech" in text_l:
            bonus += 0.05
        if name == "supplier_role_detection" and "tier" in text_l:
            bonus += 0.05
        scored.append((name, round(min(conf + bonus, 1.0), 2)))
    return sorted(scored, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Main classification
# ---------------------------------------------------------------------------

def classify_company(company: dict) -> List[Dict]:
    text = build_text(company)
    results = []
    for query_name, rule in QUERY_RULES.items():
        matched, confidence = rule.matches(text)
        if matched:
            results.append((query_name, confidence))
    results = rerank(results, text)
    output = []
    for query_name, confidence in results:
        output.append({
            "company_name": company.get("company_name", ""),
            "query": query_name,
            "confidence": confidence,
            "matched": True,
        })
    return output


def run(input_path: str = "companies.jsonl", output_path: str = "final_results.csv"):
    companies = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                companies.append(json.loads(line))

    all_results = []
    for company in companies:
        results = classify_company(company)
        if results:
            all_results.extend(results)
        else:
            all_results.append({
                "company_name": company.get("company_name", ""),
                "query": "none",
                "confidence": 0.0,
                "matched": False,
            })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["company_name", "query", "confidence", "matched"])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Processed {len(companies)} companies -> {output_path}")
    matched = sum(1 for r in all_results if r["matched"])
    print(f"Matched: {matched} / {len(all_results)} records")


if __name__ == "__main__":
    run()
