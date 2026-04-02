# Technical Writeup — Veridion Intent Qualification

## 1. Architecture

The system is built around a `QueryRule` abstraction that encodes each intent query as a declarative data structure rather than ad-hoc if/else logic. Each rule contains:

- **positive_groups**: A list of keyword groups. OR logic within each group, AND logic across groups. This enforces that a company must mention both domain vocabulary (e.g. "battery") AND context vocabulary (e.g. "electric vehicle") to match.
- **negative_signals**: Keywords that immediately disqualify a match (e.g. "lead acid" for EV battery queries).
- **confidence_boosters**: Domain-specific terms that increase confidence when present (e.g. "gigafactory", "BMS", "solid state").
- **base_confidence**: Starting confidence score, tuned per query based on signal reliability.

A **reranker** resolves companies that match multiple queries by applying small bonuses for highly specific signals and sorting by final confidence.

## 2. Query Design Rationale

### EV Battery Supply Chain
Requires both battery chemistry terms AND EV context to avoid matching generic battery companies. Confidence boosters target highly specific EV battery indicators (NCM, NCA, LFP chemistries, BMS, gigafactory).

### Fintech Growth Proxy
Requires both a fintech product signal AND a growth/funding signal. This avoids matching legacy financial institutions. Boosters reward companies that self-describe as fintech, API-first, or mention specific products like BNPL or open banking.

### Supplier Role Detection
Requires both a role signal (supplier, OEM, tier-1/2, manufacturer) AND an industry signal (automotive, aerospace, industrial). Boosters reward quality certifications (ISO 9001, IATF 16949, AS9100) which are strong supplier indicators.

## 3. Tradeoffs

| Decision | Pro | Con |
|---|---|---|
| Rule-based over ML | Interpretable, debuggable, no labeled data needed | Lower recall on edge cases |
| AND-across-groups | High precision | Misses companies with partial descriptions |
| Negative signals | Eliminates clear false positives | May introduce false negatives if negatives are too broad |
| Confidence boosters | Differentiates strong vs weak matches | Booster list requires maintenance |

## 4. Error Analysis

**False Positives** most likely occur when:
- A company description mentions EV vehicles as customers but the company is not in the supply chain.
- A fintech company mentions funding in a historical context rather than current growth.

**False Negatives** most likely occur when:
- Companies use non-standard vocabulary (e.g. "traction battery" instead of "EV battery").
- Short or sparse descriptions don't include enough keywords to trigger AND-across-groups logic.

**Mitigation**: Expanding keyword groups and adding synonym lists would improve recall. A second-pass LLM classifier could handle edge cases flagged as low-confidence.

## 5. Scaling to Production

- **Throughput**: The classifier runs in O(n * q * k) where n=companies, q=queries, k=keywords. At 1M companies and 10 queries it runs in seconds on a single CPU. Parallelization via `multiprocessing.Pool` trivially scales to billions.
- **Rule management**: QueryRules can be serialized to JSON/YAML and managed via a rule registry with versioning.
- **Monitoring**: Log confidence distributions per query. A sudden drop in match rate signals vocabulary drift in incoming data.
- **Updating rules**: New queries are added by defining a new `QueryRule` object — no code changes to the classifier logic.

## 6. Production Failure Modes

| Failure Mode | Mitigation |
|---|---|
| Description field is empty or null | `build_text()` gracefully handles missing fields with `.get()` and empty string fallbacks |
| Non-English descriptions | Extend with translation preprocessing or multilingual keyword lists |
| Keyword ambiguity (e.g. "battery" in consumer electronics) | Strengthen AND groups and add negative signals |
| Rule conflicts across queries | Reranker resolves multi-query overlaps by confidence |
| Data schema changes | Use `.get()` with defaults throughout; add schema validation at ingestion |

## 7. What I Would Do With More Time

1. **Synonym expansion** using a domain ontology (e.g. NAICS, UN SPSC codes).
2. **Active learning loop**: Use low-confidence matches as candidates for human review and model improvement.
3. **LLM-assisted rule generation**: Use an LLM to propose new keywords from a sample of matching company descriptions.
4. **Evaluation harness**: Build a labeled test set (even 50-100 manually labeled companies) to measure precision/recall per query.
5. **Confidence calibration**: Fit a Platt scaler on labeled data to map raw scores to calibrated probabilities.
