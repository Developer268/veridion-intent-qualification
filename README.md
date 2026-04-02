# Veridion Intent Qualification

A production-minded, rule-based company intent qualification engine built for the Veridion Data Challenge.

## Overview

Given a dataset of companies (`companies.jsonl`), the system classifies each company against three intent queries:

| Query | Description |
|---|---|
| `ev_battery_supply_chain` | Companies in the EV battery manufacturing/supply chain |
| `fintech_growth_proxy` | High-growth fintech companies with recent funding signals |
| `supplier_role_detection` | Tier-1/Tier-2 suppliers in automotive, aerospace, or industrial sectors |

## Architecture

- **QueryRule abstraction**: Each query is defined as a `QueryRule` with positive evidence groups (AND across groups, OR within group), negative signals, and confidence boosters.
- **Negative filtering**: Explicit negative signals disqualify false positives early.
- **Confidence scoring**: Base confidence + booster bonuses, capped at 1.0.
- **Reranker**: A lightweight reranking layer resolves ambiguous multi-query matches.

## Files

```
solution.py        # Main classifier
queries.txt        # The three intent queries
final_results.csv  # Output: company -> query matches with confidence
WRITEUP.md         # Architecture, tradeoffs, error analysis, scaling
companies.jsonl    # Input dataset (Veridion-provided)
```

## Usage

```bash
python solution.py
```

Outputs `final_results.csv` with columns: `company_name`, `query`, `confidence`, `matched`.

## Requirements

Python 3.8+, no external dependencies.

## Results Summary

The classifier processes all companies in `companies.jsonl` and outputs per-company match results with confidence scores. Companies that match no query are recorded with `matched=False`.
