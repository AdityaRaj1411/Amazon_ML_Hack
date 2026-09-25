# Business Entity Resolution — Finalized Approach

## 1. Problem framing

For every Source 1 entity, find all matching records in Source 2 / Source 3.
Source 1 is the clean, deduplicated anchor — so this is **reference matching**,
not full three-way clustering: S2 and S3 are never compared to each other,
only to S1.

Scoring is **F_0.5** (macro-averaged per S1 entity) — precision is weighted
2x over recall, and correctly predicting an empty match list (singleton)
scores a full 1.0. This shapes every downstream decision toward being
conservative about accepting a match, while still investing heavily in
**blocking recall**, since a true match dropped at the blocking stage can
never be recovered later.

---

## 2. Pipeline

```
Raw records (S1, S2, S3)
  name · address · country
        │
        ▼
┌───────────────────────────┐
│ STAGE 1 — NORMALIZATION    │
├───────────────────────────┤
│ • lowercase, unicode NFKC  │
│ • legal-suffix canonicalize│  Ltd/Limited, Pvt/Private, Corp/Corporation ...
│ • strip suffix → core name │
│ • address abbrev expansion │  Rd→Road, Opp→Opposite, No→Number ...
│ • regional-script mapping  │  hand-built dict, e.g. കേരളം → kerala
│ • pincode extraction       │  regex, country-aware
│ • country string cleanup   │  normalize casing/aliases, NOT one-hot
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ STAGE 2 — BLOCKING          │
├───────────────────────────┤
│ Union of multiple loose     │
│ passes (not one strict AND):│
│                              │
│  Pass A: country + core-name│
│          token overlap      │
│  Pass B: country + pincode  │
│          + name n-gram sim  │  (only when pincode present)
│  Pass C: country + phonetic │
│          code on core name  │  Soundex/Metaphone, catches typos
│  Fallback: name-only        │  when address/pincode missing entirely
│                              │
│ → union all passes          │
└───────────────────────────┘
        │
        ▼
  candidate_pairs.tsv
  (last-stage candidates the
   model actually scores)
        │
        ▼
┌───────────────────────────┐
│ STAGE 3 — FEATURE ENGINEER. │
├───────────────────────────┤
│ Name features:               │
│   • core seq/Levenshtein sim │
│   • Jaro-Winkler sim         │
│   • token Jaccard            │
│   • TF-IDF cosine            │
│   • suffix_match (bool/None) │
│                               │
│ Address features:            │
│   • normalized string sim    │
│   • token Jaccard            │
│   • pincode_match (bool/None)│
│   • any_address_missing flag │
│                               │
│ Country feature:             │
│   • country_exact_match      │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ STAGE 4 — CLASSIFIER         │
├───────────────────────────┤
│ Gradient-boosted trees       │
│ (LightGBM/XGBoost/CatBoost — │
│  MIT/Apache, <<8B params,    │
│  native missing-value support)│
│                               │
│ Trained on:                  │
│  + positives = ground truth  │
│  - negatives = candidate     │
│    pairs NOT in ground truth │
│    (not random pairs)        │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│ STAGE 5 — THRESHOLDING       │
├───────────────────────────┤
│ Tune decision threshold on   │
│ held-out validation split,   │
│ optimizing F_0.5 directly —  │
│ bias conservative (precision │
│ weighted 2x over recall)     │
└───────────────────────────┘
        │
        ▼
  matching_results.tsv
  (final matches, subset of
   candidate_pairs.tsv)
        │
        ▼
  utils/validate_submission.py
  → PASS before leaderboard upload
```

---

## 3. Stage details

### 3.1 Normalization
- **Name**: fold to lowercase, replace `&` → `and`, strip punctuation,
  canonicalize legal suffixes via a hand-built dictionary (`Ltd`→`Limited`,
  `Pvt`→`Private`, `Corp`→`Corporation`, etc.), separate into `core` name +
  canonical `suffix` so they can be used as independent features rather than
  one opaque string.
- **Address**: lowercase, expand common abbreviations (`Rd`→`Road`,
  `Opp`→`Opposite`, `No`→`Number`), extract PIN/postal code via regex,
  tokenize on commas/whitespace. Missing addresses (e.g. empty string)
  return an explicit `is_missing` flag rather than an empty-but-valid state,
  so downstream features can distinguish "no data" from "no similarity."
- **Regional scripts**: a small hand-built dictionary maps known non-Latin
  place names (e.g. Malayalam `കേരളം` → `kerala`) to their English
  equivalent. Built from training-data inspection, not an external
  transliteration service — kept deliberately in-house given the fair-play
  rule against external data augmentation.
- **Country**: normalize casing/whitespace only. Treated as an **open
  string set** throughout — no hardcoded `{US, India}` category list,
  since test data adds France and the private set may add more.

### 3.2 Blocking
Single strict blocking keys (e.g. `country + legal_form + postcode +
name_prefix` as one AND condition) cap recall hard, because postcode is
often missing, legal form is often absent (DBA/trade names), and prefix
blocking breaks under word-order transposition. Instead: **run several
looser blocking passes and take the union**, then measure recall of that
union against training ground truth before trusting anything downstream —
this determines the hard ceiling on final score, so it gets validated
first and independently.

### 3.3 Feature engineering
Every feature that can be legitimately unknown (suffix, pincode, address
entirely) is encoded as `None`/`NaN`, not coerced to `0` or `False`. A
missing address is not evidence of a mismatch — it's an absence of
evidence, and the classifier (via native missing-value handling in
LightGBM/XGBoost) should be allowed to lean more heavily on name+country
in that case rather than being penalized.

### 3.4 Model
Gradient-boosted trees over interpretable pairwise similarity features —
not a fine-tuned transformer — for three reasons: (1) it's easy to keep
well under the 8B-parameter / MIT-Apache license constraint, (2) it
degrades gracefully on missing features, and (3) it stays interpretable
enough to audit for the fair-play review. A small open sentence-embedding
model (e.g. MiniLM, Apache-2.0) can optionally supply one additional
semantic-similarity feature on top of the string-based ones.

Negative training examples are drawn from **candidate pairs that survived
blocking but aren't in ground truth** — not random S1×S2/S3 pairs — so the
classifier learns the actual decision boundary it will face at inference
time (near-misses), not a trivially easy one.

### 3.5 Thresholding & validation
Threshold is tuned on a held-out slice of training data, directly
optimizing macro-averaged F_0.5 (not accuracy or plain F1), since the
metric explicitly rewards correct empty predictions on singletons and
penalizes false merges 2x harder than misses.

### 3.6 Output compliance
`candidate_pairs.tsv` reflects the *last* filtering stage before the model
scores — every ID in `matching_results.tsv` must appear there. Run
`utils/validate_submission.py` locally before every leaderboard upload.

---

## 4. Fair-play boundaries observed

| Tool/technique | Status | Reasoning |
|---|---|---|
| Hand-built abbreviation/suffix dictionaries | ✅ Safe | Built from training data inspection, not external lookup |
| Regex-based address parsing | ✅ Safe | Pattern extraction, not identity lookup |
| String similarity (Levenshtein, Jaro-Winkler, TF-IDF, Jaccard) | ✅ Safe | Pure math on provided text |
| Small locally-used embedding model (e.g. MiniLM) | ✅ Safe | No live external lookups, license-compliant |
| `libpostal` / pretrained address parser | ⚠️ Ambiguous | Trained on external geographic corpus — flagged for organizer clarification, not used by default |
| Gazetteers (e.g. GeoNames) | ❌ Avoided | Functionally equivalent to the banned "geocoding/lookup" pattern |
| Geocoding APIs, lat/long distance features | ❌ Banned | Explicitly prohibited in the rules |
| Commercial ER APIs / government registry lookups | ❌ Banned | Explicitly prohibited in the rules |

---

## 5. Key decisions validated against real data

Using the worked example (`Angad Multitrade Ltd/Limited` across S1/S2/S3):
- Suffix canonicalization collapsed `Ltd` and `Limited` to an identical
  normalized name across all three records.
- Token-Jaccard address similarity stayed high (~0.87) despite an injected
  extra token (`NO. 439`) in S2 that has no counterpart in S1 — confirming
  token-set comparison is more robust than full-string comparison to
  insertion noise.
- The regional-script dictionary correctly resolved `കേരളം` → `kerala`,
  without which S1 and S2 would share zero characters on the state field.
- S3's empty address was handled via the `is_missing` flag rather than
  being scored as a mismatch, preserving the name+country match signal.
