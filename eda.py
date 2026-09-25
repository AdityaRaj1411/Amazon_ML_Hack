"""
Amazon ML Challenge 2026 — Full EDA
====================================
Runs exploratory data analysis across all 7 dataset files and saves every
chart to  eda_output/  as PNG files.

Usage (from the 'Amazon ML' folder, with the venv active):
    python eda.py

Output folder: eda_output/
"""

import os
import re
import sys
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE, "student_resource", "dataset", "train")
TEST_DIR  = os.path.join(BASE, "student_resource", "dataset", "test")
OUT_DIR   = os.path.join(BASE, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "white"})

SAMPLE_N = 200_000   # rows to sample for slow/memory-heavy analyses


# ══════════════════════════════════════════════════════════════════════════
# 1. LOADERS
# ══════════════════════════════════════════════════════════════════════════

def load_tsv(path: str) -> pd.DataFrame:
    """Read a TSV with the correct separator, string dtype, and no NaN coercion."""
    print(f"  Loading {os.path.basename(path)} …", end=" ", flush=True)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    print(f"{len(df):,} rows")
    return df


print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
tr_s1 = load_tsv(os.path.join(TRAIN_DIR, "train_source1.tsv"))
tr_s2 = load_tsv(os.path.join(TRAIN_DIR, "train_source2.tsv"))
tr_s3 = load_tsv(os.path.join(TRAIN_DIR, "train_source3.tsv"))
gt    = load_tsv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))
te_s1 = load_tsv(os.path.join(TEST_DIR,  "test_source1.tsv"))
te_s2 = load_tsv(os.path.join(TEST_DIR,  "test_source2.tsv"))
te_s3 = load_tsv(os.path.join(TEST_DIR,  "test_source3.tsv"))

sources = {
    "Train S1": tr_s1, "Train S2": tr_s2, "Train S3": tr_s3,
    "Test S1":  te_s1, "Test S2":  te_s2, "Test S3":  te_s3,
}


# ══════════════════════════════════════════════════════════════════════════
# 2. SECTION 1 — DATASET SIZES
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 1 — DATASET SIZES")
print("="*60)

sizes = {k: len(v) for k, v in sources.items()}
for k, n in sizes.items():
    print(f"  {k:12s}: {n:>12,} rows")

fig, ax = plt.subplots(figsize=(9, 4))
colors = ["#4C72B0"]*3 + ["#DD8452"]*3
bars = ax.bar(sizes.keys(), sizes.values(), color=colors, edgecolor="white", linewidth=0.8)
ax.bar_label(bars, labels=[f"{v/1e6:.2f}M" for v in sizes.values()], padding=4, fontsize=9)
ax.set_ylabel("Number of Records")
ax.set_title("Record Counts per Dataset File")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.set_ylim(0, max(sizes.values()) * 1.15)
# legend for colour coding
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4C72B0", label="Train"),
                   Patch(color="#DD8452", label="Test")], loc="upper right")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_dataset_sizes.png"))
plt.close(fig)
print("  → saved 01_dataset_sizes.png")


# ══════════════════════════════════════════════════════════════════════════
# 3. SECTION 2 — COUNTRY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 2 — COUNTRY DISTRIBUTION")
print("="*60)

country_data = {}
for name, df in sources.items():
    vc = df["country"].value_counts()
    country_data[name] = vc
    print(f"\n  {name}:")
    for c, n in vc.items():
        print(f"    {c:15s}: {n:>10,}  ({n/len(df):.1%})")

all_countries = sorted({c for vc in country_data.values() for c in vc.index})
n_src = len(sources)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
for ax, (name, vc) in zip(axes, country_data.items()):
    vals  = [vc.get(c, 0) for c in all_countries]
    total = sum(vals)
    pcts  = [v/total*100 for v in vals]
    bars  = ax.bar(all_countries, pcts, color=sns.color_palette("muted", len(all_countries)),
                   edgecolor="white")
    ax.bar_label(bars, labels=[f"{p:.1f}%" for p in pcts], padding=2, fontsize=8)
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("% of records")
    ax.set_ylim(0, 110)
    ax.tick_params(axis="x", labelsize=8)
plt.suptitle("Country Distribution per File", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_country_distribution.png"), bbox_inches="tight")
plt.close(fig)
print("\n  → saved 02_country_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 4. SECTION 3 — MISSING / EMPTY FIELDS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 3 — EMPTY FIELDS")
print("="*60)

fields = ["business_name", "business_address", "country"]
miss_rows = []
for name, df in sources.items():
    for col in fields:
        if col in df.columns:
            pct = (df[col] == "").mean() * 100
            miss_rows.append({"source": name, "field": col, "empty_pct": pct})
            if pct > 0:
                print(f"  {name:12s}  {col:20s}  {pct:.2f}% empty")

miss_df = pd.DataFrame(miss_rows)
pivot   = miss_df.pivot(index="source", columns="field", values="empty_pct").fillna(0)

fig, ax = plt.subplots(figsize=(10, 4))
pivot.plot(kind="bar", ax=ax, edgecolor="white", width=0.7)
ax.set_ylabel("% Empty Cells")
ax.set_title("Empty Field Rate per Source File")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Field", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_missing_fields.png"), bbox_inches="tight")
plt.close(fig)
print("  → saved 03_missing_fields.png")


# ══════════════════════════════════════════════════════════════════════════
# 5. SECTION 4 — GROUND TRUTH MATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 4 — GROUND TRUTH MATCH ANALYSIS")
print("="*60)

def parse_matches(s):
    if not s.strip():
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

gt["match_list"]  = gt["matched_entity_ids"].apply(parse_matches)
gt["n_matches"]   = gt["match_list"].apply(len)
gt["n_s2"]        = gt["match_list"].apply(lambda ids: sum(1 for i in ids if i.startswith("S2-")))
gt["n_s3"]        = gt["match_list"].apply(lambda ids: sum(1 for i in ids if i.startswith("S3-")))
gt["is_singleton"]= gt["n_matches"] == 0

print(f"\n  Total S1 entities in ground truth : {len(gt):,}")
print(f"  Singletons (no matches)           : {gt['is_singleton'].sum():,}  "
      f"({gt['is_singleton'].mean():.1%})")
print(f"  Entities with ≥1 match            : {(~gt['is_singleton']).sum():,}")
print(f"  Avg matches per S1 entity         : {gt['n_matches'].mean():.3f}")
print(f"  Median matches                    : {gt['n_matches'].median():.0f}")
print(f"  Max matches                       : {gt['n_matches'].max()}")
print(f"\n  S2 matches  —  avg: {gt['n_s2'].mean():.3f}  max: {gt['n_s2'].max()}")
print(f"  S3 matches  —  avg: {gt['n_s3'].mean():.3f}  max: {gt['n_s3'].max()}")
both = ((gt["n_s2"] > 0) & (gt["n_s3"] > 0)).sum()
print(f"  Entities matching BOTH S2 and S3  : {both:,}")

# Plot: match count distribution (capped at 10 for readability)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Left — full distribution capped at 10+
cap = 10
counts = gt["n_matches"].clip(upper=cap)
vc = counts.value_counts().sort_index()
labels = [str(i) if i < cap else f"{cap}+" for i in vc.index]
axes[0].bar(labels, vc.values, color="#4C72B0", edgecolor="white")
axes[0].set_xlabel("Number of Matches per S1 Entity")
axes[0].set_ylabel("Count of S1 Entities")
axes[0].set_title("Match Count Distribution")
for bar, val in zip(axes[0].patches, vc.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vc.values)*0.005,
                 f"{val/len(gt):.1%}", ha="center", va="bottom", fontsize=7.5)

# Right — S2 vs S3 match counts
s2_vc = gt["n_s2"].clip(upper=5).value_counts().sort_index()
s3_vc = gt["n_s3"].clip(upper=5).value_counts().sort_index()
x = np.arange(6)
w = 0.35
axes[1].bar(x - w/2, [s2_vc.get(i, 0) for i in range(6)], w,
            label="S2 matches", color="#4C72B0", edgecolor="white")
axes[1].bar(x + w/2, [s3_vc.get(i, 0) for i in range(6)], w,
            label="S3 matches", color="#DD8452", edgecolor="white")
axes[1].set_xlabel("Match Count (clipped at 5)")
axes[1].set_ylabel("Count of S1 Entities")
axes[1].set_title("S2 vs S3 Match Counts")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["0","1","2","3","4","5+"])
axes[1].legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_match_distribution.png"))
plt.close(fig)
print("  → saved 04_match_distribution.png")


# ══════════════════════════════════════════════════════════════════════════
# 6. SECTION 5 — BUSINESS NAME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 5 — BUSINESS NAME ANALYSIS")
print("="*60)

def name_stats(df, label):
    lens   = df["business_name"].str.len()
    tokens = df["business_name"].str.split().str.len().fillna(0)
    print(f"\n  {label}:")
    print(f"    Length  — mean: {lens.mean():.1f}  median: {lens.median():.0f}"
          f"  min: {lens.min()}  max: {lens.max()}")
    print(f"    Tokens  — mean: {tokens.mean():.1f}  median: {tokens.median():.0f}"
          f"  max: {tokens.max()}")
    # Unicode (non-ASCII) names
    non_ascii = df["business_name"].apply(lambda x: bool(re.search(r'[^\x00-\x7F]', x))).mean()
    print(f"    Non-ASCII names: {non_ascii:.1%}")
    return lens, tokens

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (name, df) in zip(axes, sources.items()):
    lens = df["business_name"].str.len().clip(upper=150)
    ax.hist(lens, bins=60, color="#4C72B0" if "Train" in name else "#DD8452",
            edgecolor="none", alpha=0.85)
    ax.axvline(lens.median(), color="red", linestyle="--", linewidth=1.2,
               label=f"median={lens.median():.0f}")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Name Length (chars, clipped at 150)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    name_stats(df, name)

plt.suptitle("Business Name Length Distribution", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_name_length_dist.png"))
plt.close(fig)
print("\n  → saved 05_name_length_dist.png")

# Token count distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (name, df) in zip(axes, sources.items()):
    tokens = df["business_name"].str.split().str.len().fillna(0).clip(upper=15)
    vc = tokens.value_counts().sort_index()
    ax.bar(vc.index.astype(int), vc.values,
           color="#4C72B0" if "Train" in name else "#DD8452", edgecolor="white")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Token Count (clipped at 15)")
    ax.set_ylabel("Count")
plt.suptitle("Business Name Token Count Distribution", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_name_token_dist.png"))
plt.close(fig)
print("  → saved 06_name_token_dist.png")


# ══════════════════════════════════════════════════════════════════════════
# 7. SECTION 6 — ADDRESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 6 — ADDRESS ANALYSIS")
print("="*60)

def has_zip(addr):
    return bool(re.search(r'\b\d{5}(?:-\d{4})?\b', addr))

def has_pin(addr):
    return bool(re.search(r'\b\d{6}\b', addr))

for name, df in sources.items():
    us_mask    = df["country"].str.lower() == "us"
    india_mask = df["country"].str.lower() == "india"
    n_us       = us_mask.sum()
    n_india    = india_mask.sum()

    if n_us > 0:
        pct_zip = df.loc[us_mask, "business_address"].apply(has_zip).mean()
        print(f"  {name:12s}  US    ZIP coverage : {pct_zip:.1%}  ({n_us:,} US records)")
    if n_india > 0:
        pct_pin = df.loc[india_mask, "business_address"].apply(has_pin).mean()
        print(f"  {name:12s}  India PIN coverage : {pct_pin:.1%}  ({n_india:,} India records)")

# Address length distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (name, df) in zip(axes, sources.items()):
    lens = df["business_address"].str.len().clip(upper=300)
    ax.hist(lens, bins=60, color="#4C72B0" if "Train" in name else "#DD8452",
            edgecolor="none", alpha=0.85)
    ax.axvline(lens.median(), color="red", linestyle="--", linewidth=1.2,
               label=f"median={lens.median():.0f}")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Address Length (chars, clipped at 300)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
plt.suptitle("Business Address Length Distribution", fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "07_address_length_dist.png"))
plt.close(fig)
print("\n  → saved 07_address_length_dist.png")

# PIN / ZIP coverage bar chart
cov_rows = []
for name, df in sources.items():
    for country, fn, label in [("US", has_zip, "ZIP"), ("India", has_pin, "PIN")]:
        mask = df["country"].str.lower() == country.lower()
        if mask.sum() > 0:
            pct = df.loc[mask, "business_address"].apply(fn).mean() * 100
            cov_rows.append({"source": name, "country": country, "coverage": pct})

cov_df = pd.DataFrame(cov_rows)
fig, ax = plt.subplots(figsize=(11, 4))
x     = np.arange(len(cov_df))
colors = ["#4C72B0" if r["country"] == "US" else "#55A868" for _, r in cov_df.iterrows()]
bars  = ax.bar(x, cov_df["coverage"], color=colors, edgecolor="white")
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in cov_df["coverage"]], padding=3, fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{r['source']}\n({r['country']})" for _, r in cov_df.iterrows()],
    fontsize=8
)
ax.set_ylabel("% records with PIN/ZIP")
ax.set_title("ZIP (US) and PIN (India) Coverage per File")
ax.set_ylim(0, 115)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4C72B0", label="US ZIP"),
                   Patch(color="#55A868", label="India PIN")])
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "08_pin_zip_coverage.png"))
plt.close(fig)
print("  → saved 08_pin_zip_coverage.png")


# ══════════════════════════════════════════════════════════════════════════
# 8. SECTION 7 — NON-ASCII / SCRIPT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 7 — NON-ASCII / SCRIPT ANALYSIS")
print("="*60)

non_ascii_rows = []
for name, df in sources.items():
    name_na  = df["business_name"].apply(lambda x: bool(re.search(r'[^\x00-\x7F]', x))).mean()*100
    addr_na  = df["business_address"].apply(lambda x: bool(re.search(r'[^\x00-\x7F]', x))).mean()*100
    non_ascii_rows.append({"source": name, "field": "name",    "pct_non_ascii": name_na})
    non_ascii_rows.append({"source": name, "field": "address", "pct_non_ascii": addr_na})
    print(f"  {name:12s}  name: {name_na:.2f}% non-ASCII   address: {addr_na:.2f}% non-ASCII")

na_df  = pd.DataFrame(non_ascii_rows)
pivot2 = na_df.pivot(index="source", columns="field", values="pct_non_ascii").fillna(0)
fig, ax = plt.subplots(figsize=(10, 4))
pivot2.plot(kind="bar", ax=ax, edgecolor="white", width=0.65, color=["#4C72B0","#DD8452"])
ax.set_ylabel("% Non-ASCII Records")
ax.set_title("Non-ASCII Character Prevalence per File (Hindi / other scripts)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Field")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "09_non_ascii_prevalence.png"), bbox_inches="tight")
plt.close(fig)
print("  → saved 09_non_ascii_prevalence.png")


# ══════════════════════════════════════════════════════════════════════════
# 9. SECTION 8 — TOP NAME TOKENS (noise pattern proxy)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 8 — TOP 30 BUSINESS NAME TOKENS (Train S1 sample)")
print("="*60)

sample = tr_s1.sample(min(SAMPLE_N, len(tr_s1)), random_state=42)
all_tokens = []
for name in sample["business_name"]:
    tokens = re.sub(r'[^\w\s]', ' ', name.lower()).split()
    all_tokens.extend(tokens)

token_counts = Counter(all_tokens)
top30 = token_counts.most_common(30)
print("  Token           Count")
for tok, cnt in top30:
    print(f"  {tok:20s} {cnt:>8,}")

tokens_df = pd.DataFrame(top30, columns=["token", "count"])
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(tokens_df["token"][::-1], tokens_df["count"][::-1],
        color="#4C72B0", edgecolor="white")
ax.set_xlabel("Frequency")
ax.set_title(f"Top 30 Business Name Tokens (Train S1, {SAMPLE_N:,}-row sample)")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "10_top_name_tokens.png"))
plt.close(fig)
print("  → saved 10_top_name_tokens.png")


# ══════════════════════════════════════════════════════════════════════════
# 10. SECTION 9 — TRAIN vs TEST COUNTRY OVERLAP
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 9 — TRAIN vs TEST COUNTRY OVERLAP")
print("="*60)

train_countries = set(pd.concat([tr_s1, tr_s2, tr_s3])["country"].unique())
test_countries  = set(pd.concat([te_s1, te_s2, te_s3])["country"].unique())
new_in_test     = test_countries - train_countries

print(f"  Countries in train : {sorted(train_countries)}")
print(f"  Countries in test  : {sorted(test_countries)}")
print(f"  NEW in test only   : {sorted(new_in_test)}  ← zero training signal for these!")

# Bar chart: record counts per country, train vs test combined
all_train = pd.concat([tr_s1[["country"]], tr_s2[["country"]], tr_s3[["country"]]])
all_test  = pd.concat([te_s1[["country"]], te_s2[["country"]], te_s3[["country"]]])

train_vc = all_train["country"].value_counts()
test_vc  = all_test["country"].value_counts()

all_ctry = sorted(set(train_vc.index) | set(test_vc.index))
x = np.arange(len(all_ctry))
w = 0.35

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w/2, [train_vc.get(c, 0) for c in all_ctry], w,
       label="Train", color="#4C72B0", edgecolor="white")
ax.bar(x + w/2, [test_vc.get(c, 0)  for c in all_ctry], w,
       label="Test",  color="#DD8452", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(all_ctry)
ax.set_ylabel("Record Count")
ax.set_title("Record Count per Country (Train vs Test)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "11_country_train_vs_test.png"))
plt.close(fig)
print("  → saved 11_country_train_vs_test.png")


# ══════════════════════════════════════════════════════════════════════════
# 11. SECTION 10 — SINGLETON vs MATCHED ENTITY PROFILE
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 10 — SINGLETON vs MATCHED ENTITY PROFILE")
print("="*60)

s1_gt = tr_s1.merge(gt[["source1_entity_id","n_matches","is_singleton"]],
                    left_on="entity_id", right_on="source1_entity_id", how="left")
s1_gt["is_singleton"] = s1_gt["is_singleton"].fillna(True)
s1_gt["name_len"]    = s1_gt["business_name"].str.len()
s1_gt["addr_len"]    = s1_gt["business_address"].str.len()
s1_gt["name_tokens"] = s1_gt["business_name"].str.split().str.len().fillna(0)

print(f"\n  Matched entities  — avg name length: {s1_gt[~s1_gt['is_singleton']]['name_len'].mean():.1f} "
      f"  avg address length: {s1_gt[~s1_gt['is_singleton']]['addr_len'].mean():.1f}")
print(f"  Singleton entities — avg name length: {s1_gt[s1_gt['is_singleton']]['name_len'].mean():.1f} "
      f"  avg address length: {s1_gt[s1_gt['is_singleton']]['addr_len'].mean():.1f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, col, label in zip(axes,
                           ["name_len",    "addr_len"],
                           ["Name Length", "Address Length"]):
    matched  = s1_gt[~s1_gt["is_singleton"]][col].clip(upper=200)
    singletons = s1_gt[s1_gt["is_singleton"]][col].clip(upper=200)
    ax.hist(matched,   bins=60, alpha=0.65, color="#4C72B0", label="Matched",   density=True)
    ax.hist(singletons,bins=60, alpha=0.65, color="#DD8452", label="Singleton", density=True)
    ax.set_xlabel(f"{label} (clipped at 200)")
    ax.set_ylabel("Density")
    ax.set_title(f"{label}: Matched vs Singleton")
    ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "12_singleton_vs_matched_profile.png"))
plt.close(fig)
print("  → saved 12_singleton_vs_matched_profile.png")


# ══════════════════════════════════════════════════════════════════════════
# 12. SECTION 11 — MATCHED PAIR NAME SIMILARITY (sampled)
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 11 — NAME SIMILARITY IN MATCHED PAIRS (sample)")
print("="*60)

try:
    from rapidfuzz import fuzz

    s2_idx = tr_s2.set_index("entity_id")["business_name"].to_dict()
    s3_idx = tr_s3.set_index("entity_id")["business_name"].to_dict()
    s1_idx = tr_s1.set_index("entity_id")["business_name"].to_dict()

    matched_gt = gt[gt["n_matches"] > 0].sample(min(5000, (gt["n_matches"]>0).sum()), random_state=42)

    sims = []
    for _, row in tqdm(matched_gt.iterrows(), total=len(matched_gt),
                       desc="  Computing name similarities", ncols=80):
        s1_name = s1_idx.get(row["source1_entity_id"], "")
        for mid in row["match_list"][:3]:   # up to 3 matches per entity
            sx_name = s2_idx.get(mid) or s3_idx.get(mid) or ""
            if s1_name and sx_name:
                sims.append(fuzz.token_set_ratio(s1_name.lower(), sx_name.lower()))

    sims = np.array(sims)
    print(f"\n  Matched pairs sampled : {len(sims):,}")
    print(f"  Name similarity (token_set_ratio):")
    print(f"    mean   : {sims.mean():.1f}")
    print(f"    median : {np.median(sims):.1f}")
    print(f"    < 50   : {(sims < 50).mean():.1%}  (hard cases)")
    print(f"    ≥ 80   : {(sims >= 80).mean():.1%}  (easy cases)")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(sims, bins=50, color="#4C72B0", edgecolor="white")
    ax.axvline(np.median(sims), color="red", linestyle="--",
               label=f"median={np.median(sims):.0f}")
    ax.set_xlabel("Token Set Ratio (0–100)")
    ax.set_ylabel("Count")
    ax.set_title("Name Similarity of True Matched Pairs (sampled)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "13_matched_name_similarity.png"))
    plt.close(fig)
    print("  → saved 13_matched_name_similarity.png")

except Exception as e:
    print(f"  Skipped (error: {e})")


# ══════════════════════════════════════════════════════════════════════════
# 13. SECTION 12 — ENTITY ID PREFIX INTEGRITY CHECK
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("SECTION 12 — ENTITY ID INTEGRITY")
print("="*60)

expected_prefix = {
    "Train S1": "S1-", "Train S2": "S2-", "Train S3": "S3-",
    "Test S1":  "S1-", "Test S2":  "S2-", "Test S3":  "S3-",
}
for name, df in sources.items():
    prefix = expected_prefix[name]
    wrong  = (~df["entity_id"].str.startswith(prefix)).sum()
    dupes  = df["entity_id"].duplicated().sum()
    status = "✓" if wrong == 0 and dupes == 0 else "✗"
    print(f"  {status} {name:12s}  wrong prefix: {wrong}  duplicate IDs: {dupes}")

# Ground truth self-match check
all_match_ids = []
for ids in gt["match_list"]:
    all_match_ids.extend(ids)
s1_in_gt = sum(1 for i in all_match_ids if i.startswith("S1-"))
print(f"\n  S1- IDs appearing as match targets in ground truth: {s1_in_gt}  (should be 0)")


# ══════════════════════════════════════════════════════════════════════════
# 14. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EDA COMPLETE — files saved to:", OUT_DIR)
print("="*60)
charts = [f for f in sorted(os.listdir(OUT_DIR)) if f.endswith(".png")]
for c in charts:
    print(f"  {c}")

print(f"\n  Total charts generated: {len(charts)}")
print("\nKey takeaways to guide your pipeline:")
print("  1. Check singleton rate   → sets your precision baseline")
print("  2. Check PIN/ZIP coverage → decides primary blocking key")
print("  3. Check non-ASCII %      → drives preprocessing decisions")
print("  4. Check name similarity  → sets your matching threshold range")
print("  5. France is test-only    → country must stay an open-set feature")
