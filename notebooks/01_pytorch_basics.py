# %% [markdown]
# # 🔥 Day 1-2: PyTorch Fundamentals
#
# Work through this file top to bottom. Run each cell using **Shift+Enter**.
#
# **By the end of Day 2, you should be able to:**
# - ✅ Create and manipulate tensors
# - ✅ Understand autograd and computational graphs
# - ✅ Build a simple neural network with nn.Module
# - ✅ Run a forward pass and understand model outputs
# - ✅ Know the difference between training mode and eval mode

# %%
# Setup — Run this cell first
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# If you have low RAM (<8GB), uncomment this:
# torch.set_num_threads(2)

# %% [markdown]
# ---
# ## Section 1: Tensors — The Foundation of Everything
#
# Tensors are like NumPy arrays but can run on GPU and track gradients.
# Every ML model input, output, and parameter is a tensor.

# %%
# Creating tensors
x = torch.tensor([1.0, 2.0, 3.0])
print(f"1D tensor: {x}")
print(f"   Shape: {x.shape}, Dtype: {x.dtype}")

# 2D tensor (like a matrix / batch of data)
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
print(f"\n2D tensor:\n{matrix}")
print(f"   Shape: {matrix.shape}  (3 rows, 2 columns)")

# Common creation methods
zeros = torch.zeros(3, 4)       # 3x4 matrix of zeros
ones = torch.ones(2, 3)         # 2x3 matrix of ones
random = torch.randn(2, 5)     # 2x5 matrix, random from normal distribution
print(f"\nRandom tensor (2x5):\n{random}")

# %%
# 🔑 KEY CONCEPT: Shape matters!
# In NLP: tensors are usually (batch_size, sequence_length, hidden_dim)
# Example: 8 texts, each 128 tokens, each token has 768 features
fake_nlp_batch = torch.randn(8, 128, 768)
print(f"NLP batch shape: {fake_nlp_batch.shape}")
print(f"   Batch size: {fake_nlp_batch.shape[0]}")
print(f"   Sequence length: {fake_nlp_batch.shape[1]}")
print(f"   Hidden dimension: {fake_nlp_batch.shape[2]}")

# %% [markdown]
# ### ✏️ Exercise 1
# Create a tensor of shape `(4, 10, 512)` and print its shape.
# This represents: 4 texts, 10 tokens each, 512-dimensional embeddings.

# %%
# YOUR CODE HERE
exercise_tensor = torch.randn(4,10,512)
print(f"Exercise tensor shape: {exercise_tensor.shape}")
print(f"    Batch Size: {exercise_tensor.shape[0]}")
print(f"    Sequence Length: {exercise_tensor.shape[1]}")
print(f"    Hidden Dimension: {exercise_tensor.shape[2]}")

# %% [markdown]
# ---
# ## Section 2: Tensor Operations

# %%
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")  # Element-wise multiply (NOT dot product)
print(f"Dot product: {torch.dot(a, b)}")  # 1*4 + 2*5 + 3*6 = 32

# %% [markdown]
# ### 🔑 Sigmoid vs Softmax — Critical for our project!
#
# - **Sigmoid**: Each score is independent (0 to 1). Text CAN be toxic AND insulting AND obscene.
# - **Softmax**: Scores sum to 1. Pick ONE class only.
#
# **Our toxicity model uses SIGMOID** because it's multi-label classification.

# %%
# Sigmoid squashes any number into 0.0 to 1.0 range
raw_scores = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
probabilities = torch.sigmoid(raw_scores)
print(f"Raw logits:         {raw_scores}")
print(f"After sigmoid:      {probabilities}")
print("Notice: negative → near 0, positive → near 1, 0 → exactly 0.5")

# Compare sigmoid vs softmax
softmax_out = F.softmax(raw_scores, dim=0)
sigmoid_out = torch.sigmoid(raw_scores)
print(f"\nSoftmax (sums to 1): {softmax_out} → sum = {softmax_out.sum():.4f}")
print(f"Sigmoid (independent): {sigmoid_out} → sum = {sigmoid_out.sum():.4f}")

# %% [markdown]
# ### ✏️ Exercise 2
# Given model logits `[-1.2, 0.8, 2.5, -0.3, 0.1, 1.9]`,
# apply sigmoid and determine which categories would be flagged at threshold=0.5

# %%
# YOUR CODE HERE
logits = torch.tensor([-1.2, 0.8, 2.5, -0.3, 0.1, 1.9])
probs = torch.sigmoid(logits)
flagged = probs >= 0.5  #True where prob >= 0.5
print(f"Probabilities: {probs}")
print(f"Flagged (>=0.5): {flagged}")

# %% [markdown]
# ---
# ## Section 3: Autograd — How Models Learn

# %%
# requires_grad=True tells PyTorch to track operations for backpropagation
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2 + 3 * x  # y = x² + 3x
z = y.sum()          # Need a scalar for backward()

print(f"x = {x}")
print(f"y = x² + 3x = {y}")
print(f"z = sum(y) = {z}")

# Compute gradients (derivatives)
z.backward()
print(f"Gradients (dy/dx = 2x + 3): {x.grad}")
# For x=2: 2*2+3=7, for x=3: 2*3+3=9 ✓

# %%
# 🔑 KEY CONCEPT: torch.no_grad()
# During INFERENCE (prediction), we don't need gradients — saves memory and speed
# This is what we'll use in our content moderator!

with torch.no_grad():
    result = x ** 2 + 3 * x
    print(f"With no_grad: {result}")
    print(f"Has grad_fn: {result.grad_fn}")  # None — no tracking!

# %% [markdown]
# ---
# ## Section 4: nn.Module — Building Neural Networks
#
# This is how ALL PyTorch models are built — including XLM-RoBERTa.

# %%
class SimpleToxicityClassifier(nn.Module):
    """
    A basic neural network for text classification.
    In reality, XLM-RoBERTa is MUCH more complex, but the interface is the same.
    """

    def __init__(self, vocab_size=30000, embedding_dim=128, num_labels=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)       # (batch, seq, embed_dim)
        x = x.mean(dim=1)                    # (batch, embed_dim) — simple pooling
        x = F.relu(self.fc1(x))             # (batch, 64)
        x = self.dropout(x)
        logits = self.fc2(x)                 # (batch, num_labels)
        return logits


# Create model and inspect it
model = SimpleToxicityClassifier()
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# %%
# Simulate a forward pass
fake_input = torch.randint(0, 30000, (2, 50))  # 2 texts, 50 tokens each
print(f"Input shape: {fake_input.shape}")

logits = model(fake_input)
print(f"Output logits shape: {logits.shape}")  # (2, 6) — 2 texts, 6 categories
print(f"Raw logits:\n{logits}")

# Convert to probabilities
probs = torch.sigmoid(logits)
print(f"\nProbabilities (after sigmoid):\n{probs}")

# 🔑 model.eval() vs model.train()
model.eval()   # Disable dropout — use for INFERENCE
model.train()  # Enable dropout — use for TRAINING

# %% [markdown]
# ---
# ## Section 5: Device Management (CPU vs GPU)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move model and data to device
model = model.to(device)
fake_input = fake_input.to(device)

with torch.no_grad():
    output = model(fake_input)
    print(f"Output on {device}: {output.shape}")

# Move back to CPU for further processing
output_cpu = output.cpu().numpy()
print(f"As numpy array:\n{output_cpu}")

# %% [markdown]
# ---
# ## ✅ Day 1-2 Checklist
#
# - [ ] What is a tensor and how is it different from a NumPy array?
# - [ ] What does `.shape` tell you? What does each dimension represent in NLP?
# - [ ] What's the difference between sigmoid and softmax? Why sigmoid for toxicity?
# - [ ] What does `requires_grad=True` do?
# - [ ] Why do we use `torch.no_grad()` during inference?
# - [ ] What does `nn.Module.forward()` do?
# - [ ] What's the difference between `model.eval()` and `model.train()`?
# - [ ] How do you move a model/tensor between CPU and GPU?
#
# **All checked? Move to `02_huggingface_intro.py`! 🚀**
#
# ```
# git add . && git commit -m "Day 1-2: PyTorch basics complete"
# ```

# %%
