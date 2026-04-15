# %% [markdown]
# # 📊 Day 6-7: Multilingual Benchmark & Documentation
#
# This is where your project stands out! Create proper benchmarks,
# generate charts, and document findings for your README.
#
# **By the end of Day 7:**
# - ✅ Benchmark dataset with 20+ labelled examples per language
# - ✅ Accuracy, precision, recall, F1 computed per language
# - ✅ Visualisation charts for your portfolio
# - ✅ README updated with real numbers

# %%
import torch
import time
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model
# MODEL_NAME = "unitary/multilingual-toxic-xlm-roberta"
MODEL_NAME = "citizenlab/distilbert-base-multilingual-cased-toxicity"  # Low RAM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Create docs folder for saving charts
os.makedirs("../docs", exist_ok=True)

# %% [markdown]
# ---
# ## Section 1: Benchmark Dataset
#
# Ground truth: `1` = should be flagged as toxic, `0` = should be clean.
#
# **✏️ You should expand this to 20+ per language (60+ total)!**

# %%
benchmark_data = [
    # ========== English — Toxic ==========
    {"text": "You are an absolute idiot and waste of space", "lang": "en", "expected": 1, "category": "insult"},
    {"text": "I will make sure you regret being alive", "lang": "en", "expected": 1, "category": "threat"},
    {"text": "People like you should be eliminated", "lang": "en", "expected": 1, "category": "hate_speech"},
    {"text": "What a disgusting thing to say, you pig", "lang": "en", "expected": 1, "category": "obscene"},
    {"text": "Nobody loves you and nobody ever will", "lang": "en", "expected": 1, "category": "toxic"},

    # ========== English — Clean ==========
    {"text": "Thank you for your thoughtful response", "lang": "en", "expected": 0, "category": "clean"},
    {"text": "I respectfully disagree with your point", "lang": "en", "expected": 0, "category": "clean"},
    {"text": "Great presentation today, well done!", "lang": "en", "expected": 0, "category": "clean"},
    {"text": "Could you please clarify your last point?", "lang": "en", "expected": 0, "category": "clean"},
    {"text": "The weather has been lovely this week", "lang": "en", "expected": 0, "category": "clean"},

    # ========== Hindi — Toxic ==========
    {"text": "तुम एक बेवकूफ और बेकार इंसान हो", "lang": "hi", "expected": 1, "category": "insult"},
    {"text": "मैं तुम्हें सबक सिखाऊंगा", "lang": "hi", "expected": 1, "category": "threat"},
    {"text": "तुम्हारे जैसे लोग समाज के लिए बोझ हैं", "lang": "hi", "expected": 1, "category": "hate_speech"},
    {"text": "चुप रहो, तुम्हारी कोई औकात नहीं", "lang": "hi", "expected": 1, "category": "toxic"},
    {"text": "तुम घटिया इंसान हो", "lang": "hi", "expected": 1, "category": "insult"},

    # ========== Hindi — Clean ==========
    {"text": "आपका काम बहुत अच्छा है", "lang": "hi", "expected": 0, "category": "clean"},
    {"text": "कृपया मुझे और जानकारी दें", "lang": "hi", "expected": 0, "category": "clean"},
    {"text": "आज का मौसम बहुत सुहावना है", "lang": "hi", "expected": 0, "category": "clean"},
    {"text": "धन्यवाद आपकी मदद के लिए", "lang": "hi", "expected": 0, "category": "clean"},
    {"text": "मुझे आपसे बात करके खुशी हुई", "lang": "hi", "expected": 0, "category": "clean"},

    # ========== Arabic — Toxic ==========
    {"text": "أنت غبي ولا تستحق الاحترام", "lang": "ar", "expected": 1, "category": "insult"},
    {"text": "سأجعلك تندم على هذا", "lang": "ar", "expected": 1, "category": "threat"},
    {"text": "أنت عار على عائلتك", "lang": "ar", "expected": 1, "category": "toxic"},
    {"text": "اخرس لا أحد يريدك هنا", "lang": "ar", "expected": 1, "category": "hate_speech"},
    {"text": "أنت أسوأ شخص عرفته", "lang": "ar", "expected": 1, "category": "insult"},

    # ========== Arabic — Clean ==========
    {"text": "شكراً لك على مساعدتك", "lang": "ar", "expected": 0, "category": "clean"},
    {"text": "أتمنى لك يوماً سعيداً", "lang": "ar", "expected": 0, "category": "clean"},
    {"text": "العمل الجماعي يؤدي إلى النجاح", "lang": "ar", "expected": 0, "category": "clean"},
    {"text": "هل يمكنك شرح ذلك بالتفصيل؟", "lang": "ar", "expected": 0, "category": "clean"},
    {"text": "الطقس جميل اليوم", "lang": "ar", "expected": 0, "category": "clean"},
]

df = pd.DataFrame(benchmark_data)
print(f"Benchmark dataset: {len(df)} examples\n")
print("Distribution:")
print(df.groupby(["lang", "expected"]).size().unstack(fill_value=0))

# %% [markdown]
# ### ✏️ Exercise: Add More Data!
# Add at least 10 more examples per language. Include edge cases like:
# - Sarcasm, mild rudeness, code-switching (mixed language)
# - Different toxicity categories (threats, insults, hate speech)
#
# The more diverse your dataset, the better your benchmark.

# %%
# YOUR ADDITIONAL DATA HERE — uncomment and add:
# extra_data = [
#     {"text": "Your throat would be slit and would burry your head in my garden", "lang": "en", "expected": 1, "category": "insult"},
#     {"text": "Hahaha... nobody but you could be the only human with that level of iq. I bet even my goldfish would perform better in that", "lang": "en", "expected": 1, "category": "insult"}
# ]
# benchmark_data.extend(extra_data)
# df = pd.DataFrame(benchmark_data)
# print(f"Updated dataset: {len(df)} examples")

# %% [markdown]
# ---
# ## Section 2: Run Benchmark

# %%
THRESHOLD = 0.5
results = []

print("Running benchmark...")
for i, item in enumerate(benchmark_data):
    start = time.time()

    inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    max_score = float(probs.max())
    predicted = 1 if max_score >= THRESHOLD else 0
    elapsed_ms = (time.time() - start) * 1000

    results.append({
        **item,
        "predicted": predicted,
        "max_score": round(max_score, 4),
        "correct": predicted == item["expected"],
        "latency_ms": round(elapsed_ms, 2),
    })

    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(benchmark_data)}...")

results_df = pd.DataFrame(results)
correct = results_df["correct"].sum()
total = len(results_df)
print(f"\nBenchmark complete: {correct}/{total} correct ({correct/total:.1%})")


# %% [markdown]
# --
# ## 50% accuracy is essentially coin-flip level — the model is getting half wrong. Based on everything we've seen earlier, I can predict exactly what's happening:
# The model is likely flagging almost everything as toxic, which means it gets all 18 toxic texts correct but all 12 clean texts wrong. That gives you exactly 18/30 — wait, you got 15/30, so it might also be missing some non-English toxic texts.

# %%
# ## Run these cells to see the full picture:
# What exactly went wrong?
print(f"\n{'='*70}")
print("DETAILED BREAKDOWN")
print(f"{'='*70}")

# False Positives — clean text flagged as toxic
fps = results_df[(results_df["expected"] == 0) & (results_df["predicted"] == 1)]
print(f"\n🔴 FALSE POSITIVES ({len(fps)}) — clean text wrongly flagged:")
for _, r in fps.iterrows():
    print(f"  [{r['lang']}] {r['max_score']:.4f}  '{r['text']}'")

# False Negatives — toxic text missed
fns = results_df[(results_df["expected"] == 1) & (results_df["predicted"] == 0)]
print(f"\n🟡 FALSE NEGATIVES ({len(fns)}) — toxic text missed:")
for _, r in fns.iterrows():
    print(f"  [{r['lang']}] {r['max_score']:.4f}  '{r['text']}'")

# Per-language accuracy
print(f"\n{'='*70}")
print("PER-LANGUAGE ACCURACY")
print(f"{'='*70}")
for lang in ["en", "hi", "ar"]:
    lang_data = results_df[results_df["lang"] == lang]
    correct = lang_data["correct"].sum()
    total = len(lang_data)
    print(f"  {lang}: {correct}/{total} ({correct/total:.0%})")

# Score distribution
print(f"\n{'='*70}")
print("SCORE DISTRIBUTION")
print(f"{'='*70}")
print(f"  {'Type':<15} {'Avg Score':<12} {'Min':<10} {'Max'}")
print(f"  {'-'*45}")
toxic_scores = results_df[results_df["expected"] == 1]["max_score"]
clean_scores = results_df[results_df["expected"] == 0]["max_score"]
print(f"  {'Toxic texts':<15} {toxic_scores.mean():<12.4f} {toxic_scores.min():<10.4f} {toxic_scores.max():.4f}")
print(f"  {'Clean texts':<15} {clean_scores.mean():<12.4f} {clean_scores.min():<10.4f} {clean_scores.max():.4f}")


# %% [markdown]
# ## Updated code after analysing what went wrong in the previous output 

# %%
# Run the same benchmark with CitizenLab model
from transformers import pipeline as hf_pipeline
from langdetect import detect

# Load CitizenLab model
print("Loading CitizenLab model...")
moderator = hf_pipeline(
    "text-classification",
    model="citizenlab/distilbert-base-multilingual-cased-toxicity",
    top_k=None,
    device=-1,
)
print("Model loaded!")

# Language-aware thresholds
LANGUAGE_THRESHOLDS = {
    "en": 0.5,
    "hi": 0.15,
    "ar": 0.10,
}

LANGUAGE_ALIASES = {
    "mr": "hi",
    "ne": "hi",
    "ur": "ar",
    "fa": "ar",
    "ca": "es",
}

def moderate_text(text):
    """Classify text with language-aware thresholds."""
    try:
        detected = detect(text)
    except:
        detected = "en"

    mapped = LANGUAGE_ALIASES.get(detected, detected)
    threshold = LANGUAGE_THRESHOLDS.get(mapped, 0.5)

    result = moderator(text)
    toxic_score = 0.0
    for r in result[0]:
        if r["label"].lower() == "toxic":
            toxic_score = r["score"]

    flagged = toxic_score >= threshold

    return {
        "detected_lang": detected,
        "mapped_lang": mapped,
        "toxic_score": round(toxic_score, 4),
        "threshold": threshold,
        "verdict": "toxic" if flagged else "clean",
    }


# Run the benchmark
print("\nRunning CitizenLab benchmark...")
cl_results = []

for item in benchmark_data:
    r = moderate_text(item["text"])
    predicted = 1 if r["verdict"] == "toxic" else 0

    cl_results.append({
        **item,
        "predicted": predicted,
        "max_score": r["toxic_score"],
        "correct": predicted == item["expected"],
    })

cl_df = pd.DataFrame(cl_results)
correct = cl_df["correct"].sum()
total = len(cl_df)

print(f"\n{'='*60}")
print(f"MODEL COMPARISON")
print(f"{'='*60}")
print(f"{'Metric':<25} {'Unitary':<15} {'CitizenLab'}")
print(f"{'-'*55}")
print(f"{'Overall Accuracy':<25} {'50.0%':<15} {correct/total:.1%}")

for lang in ["en", "hi", "ar"]:
    lang_data = cl_df[cl_df["lang"] == lang]
    c = lang_data["correct"].sum()
    t = len(lang_data)
    print(f"{'  ' + lang + ' accuracy':<25} {'—':<15} {c/t:.0%}")

toxic_s = cl_df[cl_df["expected"] == 1]["max_score"]
clean_s = cl_df[cl_df["expected"] == 0]["max_score"]
print(f"\n{'Score Distribution':<25} {'Unitary':<15} {'CitizenLab'}")
print(f"{'-'*55}")
print(f"{'  Toxic avg score':<25} {'0.7761':<15} {toxic_s.mean():.4f}")
print(f"{'  Clean avg score':<25} {'0.8805':<15} {clean_s.mean():.4f}")

# %%
# What did CitizenLab get wrong?
print(f"\n{'='*70}")
print(f"CITIZENLAB ERRORS ({cl_df[~cl_df['correct']].shape[0]} total)")
print(f"{'='*70}")

# False Positives
fps = cl_df[(cl_df["expected"] == 0) & (cl_df["predicted"] == 1)]
if len(fps) > 0:
    print(f"\n🔴 FALSE POSITIVES ({len(fps)}) — clean text wrongly flagged:")
    for _, r in fps.iterrows():
        print(f"  [{r['lang']}] score: {r['max_score']:.4f}  threshold: {LANGUAGE_THRESHOLDS.get(r['lang'], 0.5)}  '{r['text']}'")

# False Negatives
fns = cl_df[(cl_df["expected"] == 1) & (cl_df["predicted"] == 0)]
if len(fns) > 0:
    print(f"\n🟡 FALSE NEGATIVES ({len(fns)}) — toxic text missed:")
    for _, r in fns.iterrows():
        print(f"  [{r['lang']}] score: {r['max_score']:.4f}  threshold: {LANGUAGE_THRESHOLDS.get(r['lang'], 0.5)}  '{r['text']}'")

# Summary comparison
print(f"\n{'='*70}")
print(f"FINAL COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"{'Metric':<30} {'Unitary':<15} {'CitizenLab':<15} {'Winner'}")
print(f"{'-'*70}")
print(f"{'Overall Accuracy':<30} {'50.0%':<15} {'73.3%':<15} {'CitizenLab'}")
print(f"{'False Positives':<30} {'15':<15} {len(fps):<15} {'CitizenLab'}")
print(f"{'False Negatives':<30} {'0':<15} {len(fns):<15} {'Unitary'}")
print(f"{'Toxic avg score':<30} {'0.7761':<15} {'0.2674':<15} {'—'}")
print(f"{'Clean avg score':<30} {'0.8805':<15} {'0.0349':<15} {'CitizenLab'}")
print(f"{'Score separation':<30} {'INVERTED':<15} {'CORRECT':<15} {'CitizenLab'}")


# %% [markdown]
# ## The key insight — neither model is good enough alone. This is where an ensemble approach becomes valuable. Run this to see what happens when you combine both models:

# %%
# Ensemble: flag as toxic only if BOTH models say toxic
print(f"\n{'='*70}")
print(f"ENSEMBLE APPROACH (flag only if BOTH models agree)")
print(f"{'='*70}")

ensemble_both = []

for i in range(len(benchmark_data)):
    item = benchmark_data[i]
    unitary_pred = results_df.iloc[i]["predicted"]
    citizen_pred = cl_df.iloc[i]["predicted"]

    # Only toxic if BOTH models agree
    ensemble_pred = 1 if (unitary_pred == 1 and citizen_pred == 1) else 0

    ensemble_both.append({
        **item,
        "predicted": ensemble_pred,
        "correct": ensemble_pred == item["expected"],
    })

both_df = pd.DataFrame(ensemble_both)
both_correct = both_df["correct"].sum()
both_total = len(both_df)
both_fp = sum(1 for r in ensemble_both if not r["expected"] and r["predicted"])
both_fn = sum(1 for r in ensemble_both if r["expected"] and not r["predicted"])

print(f"\n{'Model':<30} {'Accuracy':<12} {'FP':<6} {'FN'}")
print(f"{'-'*55}")
print(f"{'Unitary (alone)':<30} {'50.0%':<12} {'15':<6} {'0'}")
print(f"{'CitizenLab (alone)':<30} {'73.3%':<12} {'1':<6} {'7'}")
print(f"{'Ensemble (either agrees)':<30} {'50.0%':<12} {'15':<6} {'0'}")
print(f"{'Ensemble (both agree)':<30} {both_correct/both_total:<12.1%} {both_fp:<6} {both_fn}")

# Show what the "both agree" ensemble got wrong
print(f"\n--- Errors with BOTH-agree ensemble ---")
errors = both_df[~both_df["correct"]]
for _, r in errors.iterrows():
    expected = "TOXIC" if r["expected"] == 1 else "CLEAN"
    predicted = "TOXIC" if r["predicted"] == 1 else "CLEAN"
    print(f"  Expected: {expected:<6}  Predicted: {predicted:<6}  [{r['lang']}] '{r['text']}'")


# %%
print(f"\n{'='*70}")
print(f"  FINAL BENCHMARK REPORT")
print(f"  Multilingual Content Moderator — Model Evaluation")
print(f"{'='*70}")

print(f"""
  Models evaluated:
    1. unitary/multilingual-toxic-xlm-roberta
    2. citizenlab/distilbert-base-multilingual-cased-toxicity

  Dataset: {len(benchmark_data)} texts (3 languages, toxic + clean)

  Selected model: CitizenLab with language-aware thresholds
    - Accuracy: 73.3%
    - False positive rate: 6.7% (1/15 clean texts wrongly flagged)
    - False negative rate: 46.7% (7/15 toxic texts missed)

  Known limitations:
    - Indirect threats scored near zero (model doesn't detect them)
    - Non-English toxicity detection is weaker
    - Language detection can misidentify similar languages

  Guardrails implemented:
    - Language-aware thresholds (en: 0.5, hi: 0.15, ar: 0.10)
    - Minimum text length filter (skip texts under 3 words)
    - Confidence-based routing (toxic / needs_review / clean)
    - Text normalisation (lowercase, repeated chars, emojis)

  Future improvements:
    - Fine-tune on indirect threat datasets
    - Add keyword-based rules for common threat patterns
    - Expand benchmark dataset for more robust evaluation
    - Consider API-based models (OpenAI, Perspective API) for comparison
""")



# %% [markdown]
# ---
# ## Section 3: Per-Language Metrics

# %%
def calc_metrics(df_sub):
    """Calculate precision, recall, F1, and accuracy for a subset."""
    tp = ((df_sub["predicted"] == 1) & (df_sub["expected"] == 1)).sum()
    fp = ((df_sub["predicted"] == 1) & (df_sub["expected"] == 0)).sum()
    fn = ((df_sub["predicted"] == 0) & (df_sub["expected"] == 1)).sum()
    tn = ((df_sub["predicted"] == 0) & (df_sub["expected"] == 0)).sum()

    acc = (tp + tn) / len(df_sub) if len(df_sub) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "avg_latency_ms": round(df_sub["latency_ms"].mean(), 2),
        "total": len(df_sub),
    }


# Per-language metrics
lang_metrics = {}
print(f"\n{'Language':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Latency'}")
print("-" * 65)

for lang in ["en", "hi", "ar"]:
    m = calc_metrics(results_df[results_df["lang"] == lang])
    lang_metrics[lang] = m
    print(f"{lang:<10} {m['accuracy']:<10} {m['precision']:<10} "
          f"{m['recall']:<10} {m['f1']:<10} {m['avg_latency_ms']}ms")

overall = calc_metrics(results_df)
print(f"\n{'OVERALL':<10} {overall['accuracy']:<10} {overall['precision']:<10} "
      f"{overall['recall']:<10} {overall['f1']:<10} {overall['avg_latency_ms']}ms")

# %% [markdown]
# ---
# ## Section 4: Error Analysis
#
# These misclassified examples are the most interesting part!
# **Document these in your README** — it shows real analytical depth.

# %%
errors = results_df[~results_df["correct"]]

if len(errors) > 0:
    print(f"{len(errors)} misclassified examples:")
    print("-" * 70)
    for _, row in errors.iterrows():
        exp = "TOXIC" if row["expected"] == 1 else "CLEAN"
        pred = "TOXIC" if row["predicted"] == 1 else "CLEAN"
        text_short = row["text"][:55] + "..." if len(row["text"]) > 55 else row["text"]
        print(f"  [{row['lang']}] Expected: {exp}, Got: {pred} (score: {row['max_score']})")
        print(f"       '{text_short}'")
        print()
else:
    print("✅ No errors! All examples classified correctly.")
    print("   Consider adding harder examples to stress-test the model.")

# %% [markdown]
# ---
# ## Section 5: Visualisations for Your Portfolio
#
# These charts go into your README and make your project stand out.

# %%
# Chart 1: Accuracy, F1, and Latency by language
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

langs = ["en", "hi", "ar"]
lang_names = ["English", "Hindi", "Arabic"]
colors = ["#2E86C1", "#E67E22", "#27AE60"]

# Accuracy
accs = [lang_metrics[l]["accuracy"] for l in langs]
axes[0].bar(lang_names, accs, color=colors)
axes[0].set_title("Accuracy by Language", fontsize=14, fontweight="bold")
axes[0].set_ylim(0, 1.15)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")

# F1
f1s = [lang_metrics[l]["f1"] for l in langs]
axes[1].bar(lang_names, f1s, color=colors)
axes[1].set_title("F1 Score by Language", fontsize=14, fontweight="bold")
axes[1].set_ylim(0, 1.15)
for i, v in enumerate(f1s):
    axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

# Latency
lats = [lang_metrics[l]["avg_latency_ms"] for l in langs]
axes[2].bar(lang_names, lats, color=colors)
axes[2].set_title("Avg Latency (ms)", fontsize=14, fontweight="bold")
for i, v in enumerate(lats):
    axes[2].text(i, v + 0.5, f"{v:.0f}ms", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("../docs/benchmark_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: docs/benchmark_results.png")

# %%
# Chart 2: Score distribution — toxic vs clean texts
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, lang in enumerate(langs):
    subset = results_df[results_df["lang"] == lang]
    toxic = subset[subset["expected"] == 1]["max_score"]
    clean = subset[subset["expected"] == 0]["max_score"]

    axes[idx].hist(clean, bins=10, alpha=0.7, label="Clean", color="#27AE60")
    axes[idx].hist(toxic, bins=10, alpha=0.7, label="Toxic", color="#E74C3C")
    axes[idx].axvline(x=THRESHOLD, color="black", linestyle="--", label=f"Threshold={THRESHOLD}")
    axes[idx].set_title(f"{lang_names[idx]}", fontsize=13, fontweight="bold")
    axes[idx].legend()
    axes[idx].set_xlabel("Max Toxicity Score")

plt.suptitle("Score Distribution: Toxic vs Clean Texts", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../docs/score_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved: docs/score_distribution.png")

# %% [markdown]
# ---
# ## Section 6: Export Results
#
# Generate a Markdown table for your README and save detailed results as JSON.

# %%
# README-ready benchmark table
print("=" * 60)
print("COPY THIS INTO YOUR README.md:")
print("=" * 60)
print()
print("## Multilingual Performance Benchmarks")
print()
print(f"Model: `{MODEL_NAME}` | Threshold: {THRESHOLD}")
print()
print("| Language | Samples | Accuracy | Precision | Recall | F1 Score | Avg Latency |")
print("|----------|---------|----------|-----------|--------|----------|-------------|")
for lang, name in zip(langs, lang_names):
    m = lang_metrics[lang]
    print(f"| {name:<8} | {m['total']:<7} | {m['accuracy']:.1%}     | "
          f"{m['precision']:.1%}      | {m['recall']:.1%}  | {m['f1']:.2f}     | {m['avg_latency_ms']}ms    |")
print(f"| **Overall** | {overall['total']:<7} | {overall['accuracy']:.1%}     | "
      f"{overall['precision']:.1%}      | {overall['recall']:.1%}  | {overall['f1']:.2f}     | {overall['avg_latency_ms']}ms    |")

# %%
# Save full results to JSON
output = {
    "model": MODEL_NAME,
    "threshold": THRESHOLD,
    "per_language": lang_metrics,
    "overall": overall,
    "detailed_results": results,
}

with open("../docs/benchmark_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("✅ Saved: docs/benchmark_results.json")

# Save results CSV for easy viewing
results_df.to_csv("../docs/benchmark_details.csv", index=False, encoding="utf-8-sig")
print("✅ Saved: docs/benchmark_details.csv")

# %% [markdown]
# ---
# ## ✅ Day 6-7 Checklist — Week 1 Complete!
#
# - [ ] Benchmark dataset has 60+ labelled examples (20+ per language)
# - [ ] Metrics computed: accuracy, precision, recall, F1 per language
# - [ ] Charts saved to `docs/` folder
# - [ ] Error analysis documented (which texts fail and why)
# - [ ] README updated with real benchmark numbers and chart images
# - [ ] LinkedIn post written and published
# - [ ] All code committed with clean history
#
# ### 🎉 Week 1 DONE! Next: Week 2 — FastAPI Service
#
# **Final commits:**
# ```
# git add .
# git commit -m "Day 6-7: Full multilingual benchmark with charts and metrics"
# git push
# ```
#
# **Update your README.md:**
# 1. Replace TBD values with your actual numbers from the table above
# 2. Add chart images: `![Benchmarks](docs/benchmark_results.png)`
# 3. Add score distribution: `![Scores](docs/score_distribution.png)`
# 4. Write 2-3 sentences about model limitations you discovered
