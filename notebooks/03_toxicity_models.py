# %% [markdown]
# # 🔬 Day 4-5: Toxicity Models — Thresholds, Edge Cases & Batch Processing
#
# **By the end of Day 5:**
# - ✅ Understand how threshold tuning affects precision/recall
# - ✅ Handle edge cases (sarcasm, short text, mixed language)
# - ✅ Process batches of text efficiently
# - ✅ Build a production-ready classify() function

# %%
import torch
import time
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model (uses cache from Day 3 — should be fast now)
# If you used the lighter model on Day 3, use the same one here:
# MODEL_NAME = "unitary/multilingual-toxic-xlm-roberta"
MODEL_NAME = "citizenlab/distilbert-base-multilingual-cased-toxicity"  # Low RAM option

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")
print(f"Categories: {list(model.config.id2label.values())}")

# %% [markdown]
# ---
# ## Production-Ready classify() Function
#
# This is the function that will go into `app/models/moderator.py`.

# %%
def classify(text, model, tokenizer, threshold=0.5):
    """
    Production-ready toxicity classification.

    Args:
        text: Input text (any supported language)
        model: Loaded transformer model
        tokenizer: Corresponding tokenizer
        threshold: Flagging threshold (0.0 to 1.0)

    Returns:
        dict with categories, verdict, max_score, processing_time_ms
    """
    start = time.time()
    _device = next(model.parameters()).device

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=512, padding=True
    ).to(_device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    categories = {}
    max_score = 0.0
    for idx, (label_id, label_name) in enumerate(model.config.id2label.items()):
        score = float(probs[idx]) if probs.ndim > 0 else float(probs)
        categories[label_name] = {"score": round(score, 4), "flagged": score >= threshold}
        max_score = max(max_score, score)

    any_flagged = any(c["flagged"] for c in categories.values())
    elapsed_ms = (time.time() - start) * 1000

    return {
        "categories": categories,
        "verdict": "toxic" if any_flagged else "clean",
        "max_score": round(max_score, 4),
        "processing_time_ms": round(elapsed_ms, 2),
    }


# Quick test
result = classify("You are amazing!", model, tokenizer)
print(json.dumps(result, indent=2))

# %% [markdown]
# ---
# ## Threshold Experimentation
#
# 🔑 **This is critical!**
# - Low threshold (0.3) → catches more toxic content but more false positives
# - High threshold (0.7) → fewer false positives but might miss toxic content
# - The right threshold depends on your use case

# %%
test_text = "Shut up, nobody asked for your opinion"

print(f"Text: '{test_text}'")
print(f"\n{'Threshold':<12} {'Verdict':<10} {'Flagged Categories'}")
print("-" * 60)

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    result = classify(test_text, model, tokenizer, threshold=threshold)
    flagged = [k for k, v in result["categories"].items() if v["flagged"]]
    flagged_str = ", ".join(flagged) if flagged else "none"
    print(f"{threshold:<12} {result['verdict']:<10} {flagged_str}")

# %% [markdown]
# ### ✏️ Exercise
# Try different thresholds on the edge cases below.
# Find the "sweet spot" that correctly handles most of them.

# %%
edge_cases = [
    "Please stop talking, you're annoying",
    "I disagree with your political views",
    "You should be ashamed of yourself",
    "That's the dumbest thing I've ever heard",
    "I hope you suffer for what you did",
]

# Try threshold=0.5 first, then adjust
print("=" * 65)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print(f"\n--- Threshold: {threshold} ---")
    correct = 0
    total = len(edge_cases)
    for text in edge_cases:
        result = classify(text, model, tokenizer, threshold=0.5)
        emoji = "🚩" if result["verdict"] == "toxic" else "✅"
        print(f"{emoji} {result['max_score']:.3f}  {text}")
    print()


# %% [markdown]
# ---
# ## Edge Case Testing
#
# Test these tricky inputs — the model might struggle with some!

# %%
tricky_texts = [
    ("Very short", "hi"),
    ("Very short", "ok"),
    ("Very short", "no"),
    ("All caps", "I HATE THIS"),
    ("Lowercase", "i hate this"),
    ("Sarcasm", "Oh wow, you're SO smart"),
    ("Question form", "Are you stupid?"),
    ("Statement form", "You are stupid"),
    ("Repeated chars", "You are stuuuuuupid"),
    ("Mixed lang", "That's really बकवास"),  # English + Hindi
    ("Emoji heavy", "You're trash 🗑️🤮"),
    ("Passive aggressive", "Sure, whatever you say, genius"),
    ("Constructive criticism", "Your work needs significant improvement"),
]

print(f"{'Type':<22} {'Verdict':<8} {'Score':<8} {'Text'}")
print("-" * 75)

for label, text in tricky_texts:
    result = classify(text, model, tokenizer)
    emoji = "🚩" if result["verdict"] == "toxic" else "✅"
    print(f"{label:<22} {emoji} {result['verdict']:<5} {result['max_score']:<8.4f} {text}")


# %% [markdown]
# ---
# ## Above code gave wrong output, so its fixed in the below code!!

#%%
import re

def classify_safe(text, model, tokenizer, threshold=0.5):
    """Production-ready wrapper with all guardrails."""

    # Guard 1: Minimum text length filter
    if len(text.split()) < 3:
        return {
            "verdict": "clean",
            "max_score": 0.0,
            "categories": {},
            "note": "Text too short for reliable classification"
        }

    # Guard 2: Normalise text before classification
    normalised = text.lower()
    normalised = re.sub(r'(.)\1{2,}', r'\1\1', normalised)  # "stuuuuupid" → "stuupid"
    normalised = re.sub(r'[^\w\s]', '', normalised)          # remove emojis/special chars

    # Run the model
    result = classify(normalised, model, tokenizer, threshold=threshold)

    # Guard 3: Confidence-based routing
    score = result["max_score"]
    if score >= 0.85:
        result["verdict"] = "toxic"
        result["confidence"] = "high"
    elif score >= 0.5:
        result["verdict"] = "needs_review"
        result["confidence"] = "uncertain"
    else:
        result["verdict"] = "clean"
        result["confidence"] = "high"

    return result


tricky_texts = [
    ("Very short", "hi"),
    ("Very short", "ok"),
    ("Very short", "no"),
    ("All caps", "I HATE THIS"),
    ("Lowercase", "i hate this"),
    ("Sarcasm", "Oh wow, you're SO smart"),
    ("Question form", "Are you stupid?"),
    ("Statement form", "You are stupid"),
    ("Repeated chars", "You are stuuuuuupid"),
    ("Mixed lang", "That's really बकवास"),
    ("Emoji heavy", "You're trash 🗑️🤮"),
    ("Passive aggressive", "Sure, whatever you say, genius"),
    ("Constructive criticism", "Your work needs significant improvement"),
]

print(f"{'Type':<22} {'Verdict':<14} {'Confidence':<12} {'Score':<8} {'Text'}")
print("-" * 90)

for label, text in tricky_texts:
    result = classify_safe(text, model, tokenizer)
    
    if result["verdict"] == "toxic":
        emoji = "🚩"
    elif result["verdict"] == "needs_review":
        emoji = "⚠️"
    else:
        emoji = "✅"

    confidence = result.get("confidence", "n/a")
    note = f"  [{result['note']}]" if "note" in result else ""
    print(f"{label:<22} {emoji} {result['verdict']:<12} {confidence:<12} {result['max_score']:<8.4f} {text}{note}")



# %% [markdown]
# ---
# ## Cross-Language Performance
#
# Same meaning, three languages — does the model score them similarly?
# **Note any big differences** — this is model bias worth documenting!

# %%
multilingual_tests = [
    {"cat": "insult", "en": "You are a terrible and disgusting person",
     "hi": "तुम एक भयानक और घिनौने इंसान हो", "ar": "أنت شخص فظيع ومقرف"},
    {"cat": "threat", "en": "I will find you and make you pay",
     "hi": "मैं तुम्हें ढूंढूंगा और तुम्हें सबक सिखाऊंगा", "ar": "سأجدك وأجعلك تدفع الثمن"},
    {"cat": "clean", "en": "Thank you for your kindness and support",
     "hi": "आपकी दयालुता और समर्थन के लिए धन्यवाद", "ar": "شكراً لك على لطفك ودعمك"},
    {"cat": "clean", "en": "The weather is beautiful today",
     "hi": "आज मौसम बहुत अच्छा है", "ar": "الطقس جميل اليوم"},
]

print(f"{'Category':<10} {'Lang':<6} {'Verdict':<8} {'Max Score':<12} {'Time'}")
print("-" * 55)

for test in multilingual_tests:
    for lang in ["en", "hi", "ar"]:
        result = classify(test[lang], model, tokenizer)
        print(f"{test['cat']:<10} {lang:<6} {result['verdict']:<8} "
              f"{result['max_score']:<12} {result['processing_time_ms']:.0f}ms")
    print()

# %% [markdown]
#---
# ##The output of the above code was not upto the mark...!!

# %% [markdown]
# ---
# ## Batch Processing Performance

# %%
sample_texts = [
    "You are wonderful!", "I hate everything about you",
    "The report looks great, thanks", "You're an idiot and everyone knows it",
    "Can we schedule a meeting for tomorrow?", "Go away, nobody wants you here",
    "Excellent work on the presentation", "You should be fired immediately",
    "I appreciate your feedback", "This is the worst thing I've ever seen",
    "आपका काम बहुत अच्छा है",           # "Your work is very good"
    "तुम बेकार हो",                       # "You are useless"
    "شكراً لمساعدتك",                     # "Thanks for your help"
    "أنت لا تستحق شيئاً",                # "You deserve nothing"
    "Have a great weekend!", "You make me sick",
    "Great suggestion, let's try it", "Nobody cares what you think",
    "Looking forward to working together", "You'll regret this, mark my words",
]

start = time.time()
results = []
for text in sample_texts:
    r = classify(text, model, tokenizer)
    r["text"] = text
    results.append(r)

total_ms = (time.time() - start) * 1000
toxic_count = sum(1 for r in results if r["verdict"] == "toxic")

print(f"Processed {len(sample_texts)} texts in {total_ms:.0f}ms")
print(f"Average: {total_ms / len(sample_texts):.0f}ms per text")
print(f"Toxic: {toxic_count} | Clean: {len(sample_texts) - toxic_count}")

# %%
# Detailed results
print(f"\n{'Verdict':<8} {'Score':<8} {'Text'}")
print("-" * 65)
for r in results:
    emoji = "🚩" if r["verdict"] == "toxic" else "✅"
    text_short = r["text"][:48] + "..." if len(r["text"]) > 48 else r["text"]
    print(f"{emoji} {r['verdict']:<5} {r['max_score']:<8.4f} {text_short}")

# %% [markdown]
# ---
# ## ✅ Day 4-5 Checklist
#
# - [ ] classify() function works reliably with threshold parameter
# - [ ] Tested thresholds from 0.3 to 0.8 and understand trade-offs
# - [ ] Identified edge cases where the model struggles
# - [ ] Noted cross-language performance differences
# - [ ] Batch processing works on 20+ texts
# - [ ] Know which threshold you'll recommend as default
#
# **All checked? Move to `04_multilingual_benchmark.py`! 🚀**
#
# ```
# git add . && git commit -m "Day 4-5: Threshold tuning, edge cases, batch processing"
# ```

# %%
