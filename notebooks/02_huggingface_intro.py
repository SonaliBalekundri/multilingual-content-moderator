# %% [markdown]
# # 🤗 Day 3: Hugging Face Transformers
#
# Today you'll load a REAL toxicity detection model and classify text
# across English, Hindi, and Arabic.
#
# **By the end of Day 3:**
# - ✅ Understand what tokenizers do and why they matter
# - ✅ Load any pre-trained model from Hugging Face Hub
# - ✅ Run inference with pipeline (easy) and manually (pro)
# - ✅ Interpret model outputs (logits → probabilities → labels)

# %%
import torch
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch: {torch.__version__} | Device: {device}")

# %% [markdown]
# ---
# ## Section 1: Tokenizers — Turning Text into Numbers
#
# Models don't understand text — they understand numbers (token IDs).
# A tokenizer converts text → token IDs and back.

# %%
# Load a multilingual tokenizer (XLM-RoBERTa)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# Tokenize English text
text_en = "This content is harmful and offensive"
tokens_en = tokenizer(text_en, return_tensors="pt")

print(f"Original text: '{text_en}'")
print(f"Token IDs: {tokens_en['input_ids']}")
print(f"Token count: {tokens_en['input_ids'].shape[1]}")

# Decode back to see what each token represents
token_list = tokens_en["input_ids"][0].tolist()
decoded_tokens = [tokenizer.decode([t]) for t in token_list]
print(f"Individual tokens: {decoded_tokens}")

# %%
# 🔑 The SAME tokenizer handles multiple languages!
text_hi = "यह सामग्री हानिकारक है"  # Hindi: "This content is harmful"
text_ar = "هذا المحتوى ضار"          # Arabic: "This content is harmful"

tokens_hi = tokenizer(text_hi, return_tensors="pt")
tokens_ar = tokenizer(text_ar, return_tensors="pt")

print(f"English tokens: {tokens_en['input_ids'].shape[1]}")
print(f"Hindi tokens:   {tokens_hi['input_ids'].shape[1]}")
print(f"Arabic tokens:  {tokens_ar['input_ids'].shape[1]}")

# %% [markdown]
# ### ✏️ Exercise
# Tokenize a sentence in English, Hindi, and Arabic.
# Compare the token counts — which language uses more tokens for the same meaning?

# %%
# YOUR CODE HERE
en_text = "Be careful of toxic people!"
hi_text = "हानिकारक लोगों से सावधान रहें!"
ar_text = "احذر من الأشخاص السامين!"

en_tokens = tokenizer(en_text, return_tensors="pt")
hi_tokens = tokenizer(hi_text, return_tensors="pt")
ar_tokens = tokenizer(ar_text, return_tensors="pt")

print(f"English Tokens: {en_tokens['input_ids'].shape[1]}")
print(f"Hindi Tokens: {hi_tokens['input_ids'].shape[1]}")
print(f"Arabic Tokens: {ar_tokens['input_ids'].shape[1]}")

# %% [markdown]
# ---
# ## Section 2: Loading the Toxicity Model
#
# First download is ~1.1GB — takes 2-5 min depending on your internet.
# After that, it's cached locally and loads in ~10-15 seconds.
#
# **Low RAM option**: If your system has <8GB RAM, uncomment the alternative model below.

# %%
# ===== CHOOSE YOUR MODEL =====

# Full model (1.1GB, best accuracy — use if you have 8GB+ RAM):
MODEL_NAME = "unitary/multilingual-toxic-xlm-roberta"

# Lighter model (~500MB — use if you have <8GB RAM):
# MODEL_NAME = "citizenlab/distilbert-base-multilingual-cased-toxicity"

# ==================================

print(f"Loading model: {MODEL_NAME}")
print("First download may take a few minutes — please wait... ☕")
start = time.time()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Move to GPU if available, otherwise CPU is fine
model = model.to(device)
model.eval()  # Set to inference mode

print(f"\nLoaded in {time.time() - start:.1f}s on {device}")
print(f"Number of labels: {model.config.num_labels}")
print(f"Categories: {list(model.config.id2label.values())}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
# ---
# ## Section 3: Easy Mode — Pipeline API

# %%
classifier = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None,  # Return ALL label scores, not just top-1
    device=0 if torch.cuda.is_available() else -1,
)

# Test with English toxic text
result = classifier("You are a terrible, disgusting person!")
print("English toxic text:")
for r in result[0]:
    print(f"  {r['label']}: {r['score']:.4f}")

# Test with clean text
result = classifier("Thank you for your help, I really appreciate it!")
print("\nEnglish clean text:")
for r in result[0]:
    print(f"  {r['label']}: {r['score']:.4f}")

# Test with Hindi
result = classifier("तुम एक भयानक इंसान हो")  # "You are a terrible person"
print("\nHindi toxic text:")
for r in result[0]:
    print(f"  {r['label']}: {r['score']:.4f}")

# %% [markdown]
# ---
# ## Section 4: Pro Mode — Manual Inference
#
# This is what your `ContentModerator` class does under the hood.
# Manual mode gives you control over batch processing, thresholds, etc.

# %%
def classify_text(text: str, model, tokenizer, threshold: float = 0.5):
    """
    Classify a single text for toxicity.
    This function will eventually live in app/models/moderator.py
    """
    _device = next(model.parameters()).device

    # Step 1: Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",     # Return PyTorch tensors
        truncation=True,          # Cut off at max length
        max_length=512,           # XLM-RoBERTa max is 512 tokens
        padding=True,             # Pad shorter sequences
    ).to(_device)

    # Step 2: Run model (no gradients needed for inference)
    with torch.no_grad():
        outputs = model(**inputs)
        # Apply sigmoid for multi-label (NOT softmax)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    # Step 3: Map to category names and apply threshold
    results = {}
    for idx, (label_id, label_name) in enumerate(model.config.id2label.items()):
        score = float(probs[idx]) if probs.ndim > 0 else float(probs)
        results[label_name] = {
            "score": round(score, 4),
            "flagged": score >= threshold,
        }

    return results


# Quick test
result = classify_text("You are amazing!", model, tokenizer)
print("Clean text result:")
for cat, info in result.items():
    flag = "🚩" if info["flagged"] else "✅"
    print(f"  {cat}: {info['score']:.4f} {flag}")

#%% [markdown]
# ---
# ## There is a model error in the output of the below code

# %%
# Test across all three languages
test_texts = [
    ("English toxic", "You are stupid and worthless"),
    ("English clean", "Have a wonderful day!"),
    ("Hindi toxic", "तुम बेवकूफ हो"),        # "You are stupid"
    ("Hindi clean", "आपका दिन शुभ हो"),       # "Have a good day"
    ("Arabic toxic", "أنت غبي"),              # "You are stupid"
    ("Arabic clean", "أتمنى لك يوماً سعيداً"),  # "Have a happy day"
]

print("Multilingual classification results:")
print("-" * 60)

for label, text in test_texts:
    result = classify_text(text, model, tokenizer)
    max_cat = max(result.items(), key=lambda x: x[1]["score"])
    any_flagged = any(v["flagged"] for v in result.values())
    verdict = "🚩 TOXIC" if any_flagged else "✅ CLEAN"

    print(f"\n[{label}] '{text}'")
    print(f"  Verdict: {verdict}")
    print(f"  Top category: {max_cat[0]} ({max_cat[1]['score']:.4f})")
    flagged = {k: v for k, v in result.items() if v["flagged"]}
    if flagged:
        print(f"  Flagged: {', '.join(flagged.keys())}")

#%% [markdown]
# ---
# ## New Model to fix the errors from the previous outputs
# Let's test an alternative model that may handle Hindi/Arabic better

# %%
# Switch to CitizenLab model — much better multilingual performance
from transformers import pipeline as hf_pipeline
from langdetect import detect

print("Loading CitizenLab model...")
moderator = hf_pipeline(
    "text-classification",
    model="citizenlab/distilbert-base-multilingual-cased-toxicity",
    top_k=None,
    device=-1,
)
print("Model loaded!")

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


# Test all 6
test_texts = [
    ("English toxic", "You are stupid and worthless"),
    ("Hindi toxic", "तुम बेवकूफ हो"),
    ("Arabic toxic", "أنت غبي"),
    ("English clean", "Have a wonderful day!"),
    ("Hindi clean", "आपका दिन शुभ हो"),
    ("Arabic clean", "أتمنى لك يوماً سعيداً"),
]

print(f"\n{'Label':<18} {'Lang':<6} {'Score':<10} {'Threshold':<10} {'Verdict'}")
print("-" * 60)

for label, text in test_texts:
    r = moderate_text(text)
    emoji = "🚩" if r["verdict"] == "toxic" else "✅"
    print(f"{label:<18} {r['mapped_lang']:<6} {r['toxic_score']:<10} {r['threshold']:<10} {emoji} {r['verdict'].upper()}")

# %% [markdown]
# ---
# ## PROJECT UPDATE ➖

# After enquiring the error in the output for Hindi and Arabic, I evaluated two multilingual toxicity models. The XLM-RoBERTa model showed severe cross-lingual bias — identical insults scored 0.99 in English but 0.06 in Hindi. I switched to a DistilBERT multilingual model and implemented per-language threshold calibration, achieving correct classification across all three languages.
# **What changed:**
# - **Model switched** from `unitary/multilingual-toxic-xlm-roberta` to `citizenlab/distilbert-base-multilingual-cased-toxicity` — better cross-lingual performance, smaller size
# - **Per-language thresholds** built into config: English 0.5, Hindi 0.15, Arabic 0.10
# - **Language alias mapping** added: Marathi → Hindi, Nepali → Hindi, Urdu → Arabic, Farsi → Arabic
# - **Moderator** now auto-detects language and applies the correct threshold


# %% [markdown]
# ---
# ## Section 5: Understanding Model Outputs

# %%
text = "I will hurt you and your family"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

with torch.no_grad():
    outputs = model(**inputs)

print(f"Text: '{text}'")
print(f"\nRaw logits (before sigmoid): {outputs.logits}")
print(f"Logits shape: {outputs.logits.shape}")

probs = torch.sigmoid(outputs.logits)
print(f"\nProbabilities (after sigmoid): {probs}")

# 🔑 Why sigmoid and not softmax?
# Because a text can be BOTH toxic AND a threat AND insulting.
# Softmax would force the scores to sum to 1, implying only ONE category.
# Sigmoid lets each category score independently.
print(f"\nSum of probabilities: {probs.sum().item():.4f}")
print("(Notice it doesn't sum to 1 — each category is independent!)")

# %% [markdown]
# ### ✏️ Exercise
# Pick 5 texts in different languages (can be from social media, comments, etc.).
# Run them through `classify_text()` and see if the model catches them correctly.
# Note any cases where the model gets it wrong — useful for benchmarking on Day 6-7.

# %%
# YOUR CODE HERE
# my_texts = [
#     ("description", "your text here"),
# ]
# for label, text in my_texts:
#     result = classify_text(text, model, tokenizer)
#     ...
my_texts = [
    ("Spanish toxic", "Eres un idiota inútil"),              # "You're a useless idiot"
    ("French clean", "Bonjour, comment allez-vous?"),        # "Hello, how are you?"
    ("Hindi sarcastic", "वाह, क्या बुद्धि है तुम्हारी"),      # "Wow, what intelligence you have" (sarcasm)
    ("Arabic toxic", "أنت أحمق ولا تستحق الاحترام"),         # "You're a fool and don't deserve respect"
    ("English subtle", "Nobody asked for your opinion, just shut up"),  # Toxic but no slurs
    ("Code-mixed", "You are such a bewakoof, get lost yaar"),  # Hindi-English mix
    ("English edge", "I'll destroy you in this game tonight"),  # Gaming context, not a real threat
]

print(f"\n{'Label':<20} {'Verdict':<10} {'Top Score':<12} {'Flagged Categories'}")
print("-" * 75)

for label, text in my_texts:
    result = classify_text(text, model, tokenizer)
    max_cat = max(result.items(), key=lambda x: x[1]["score"])
    any_flagged = any(v["flagged"] for v in result.values())
    flagged_cats = [k for k, v in result.items() if v["flagged"]]
    verdict = "🚩 TOXIC" if any_flagged else "✅ CLEAN"

    print(f"{label:<20} {verdict:<10} {max_cat[0]}: {max_cat[1]['score']:<8.4f} {', '.join(flagged_cats) if flagged_cats else 'none'}")

# Also run through CitizenLab model for comparison
print("\n\n📊 COMPARISON: CitizenLab model (language-aware thresholds)")
print(f"\n{'Label':<20} {'Lang':<6} {'Score':<10} {'Threshold':<10} {'Verdict'}")
print("-" * 65)

for label, text in my_texts:
    r = moderate_text(text)
    emoji = "🚩" if r["verdict"] == "toxic" else "✅"
    print(f"{label:<20} {r['mapped_lang']:<6} {r['toxic_score']:<10.4f} {r['threshold']:<10} {emoji} {r['verdict'].upper()}")



# %% [markdown]
# ---
# ## ✅ Day 3 Checklist
#
# - [ ] What does a tokenizer do? Why is it paired with a specific model?
# - [ ] What's the difference between pipeline() and manual inference?
# - [ ] Why do we use sigmoid instead of softmax for toxicity detection?
# - [ ] What is truncation and why is max_length=512 important?
# - [ ] What does model.eval() do and why is it needed?
# - [ ] Can you explain the flow: text → tokens → model → logits → sigmoid → scores?
#
# **All checked? Move to `03_toxicity_models.py`! 🚀**
#
# ```
# git add . && git commit -m "Day 3: HuggingFace intro, model loaded, inference working"
# ```
