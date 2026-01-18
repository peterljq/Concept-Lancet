from sparse_coding import decompose
import torch

# Demo case 1: Simple diagonal matrix decomposition
print("\n=== Demo Case 1: Simple Diagonal Matrix Decomposition ===\n")

# target matrix
representation = torch.diag(torch.tensor([0, 5, 2, 1, 78, 16]))

dictionary = [torch.diag(torch.tensor([1,0,0,0,0,0])), torch.diag(torch.tensor([0,1,0,0,0,0])), torch.diag(torch.tensor([0,0,1,0,0,0])), torch.diag(torch.tensor([0,0,0,1,0,0])), torch.diag(torch.tensor([0,0,0,0,1,0])), torch.diag(torch.tensor([0,0,0,0,0,1]))]

decomposed_coefficient = decompose(representation, dictionary, tau = 0.99, alpha = 0.01, print_level = 0, normalize = False, return_rectr_err = False)

# Validate the compositional analysis
print(decomposed_coefficient)

# Demo case 2: CLIP embedding decomposition
print("\n=== Demo Case 2: CLIP Embedding Decomposition ===\n")

import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

target_phrase = "a very cute dog"
dict_words = ["dog", "pet", "plant", "cat", "work", "cute", "small", "car", "building", "large", "ice", "laptop", "drink", "coffee"]

# Get CLIP embeddings for target phrase
with torch.no_grad():
    target_text_tokens = clip.tokenize(target_phrase).to(device)
    target_embedding = model.encode_text(target_text_tokens)
    target_embedding = target_embedding / target_embedding.norm(dim=-1, keepdim=True)

print(f"Target phrase: '{target_phrase}'")
print(f"Target embedding shape: {target_embedding.shape}")
print(f"Dictionary size: {len(dict_words)}")

# Get CLIP embeddings for dictionary words
dictionary_embeddings = []
with torch.no_grad():
    for word in dict_words:
        word_tokens = clip.tokenize(word).to(device)
        word_embedding = model.encode_text(word_tokens)
        word_embedding = word_embedding / word_embedding.norm(dim=-1, keepdim=True)
        dictionary_embeddings.append(word_embedding.squeeze(0).cpu())

# Decompose target embedding using dictionary
decomposed_coeff = decompose(
    target_embedding.squeeze(0).cpu(), 
    dictionary_embeddings, 
    tau=0.95, 
    alpha=0.05, 
    print_level=0, 
    normalize=True, 
    return_rectr_err=False
)

print(f"\nDecomposed coefficients: {decomposed_coeff}")

# Validate top contributing words
top_indices = torch.argsort(torch.abs(decomposed_coeff), descending=True)[:2]
print(f"\nTop 2 contributing words:")
for idx in top_indices:
    print(f"  {dict_words[idx]}: {decomposed_coeff[idx]:.4f}")