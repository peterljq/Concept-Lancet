# Concept Lancet: Image Editing with Compositional Representation Transplant (CVPR 2025)
Concept Lancet (CoLan) enables interpretable zero-shot representation manipulation for diffusion-based image editing by task-specific decomposition and transplant of visual concepts. This repository provides the CoLan-150K stimulus dataset and the decomposition procedure used in the paper.

## CoLan-150K Stimulus Dataset
The dataset includes:

- **concept_list.txt**: The full list of concepts in the dataset.
- **concept_stimulus.zip**: All stimuli organized by concept for concept decomposition and analysis

## Concept Decomposition via Sparse Coding
We provide the code for sparse coding-based decomposition of concepts. The `decompose()` function enables zero-shot decomposition of target representations (matrices or embeddings) into sparse linear combinations of dictionary atoms using elastic net regularization.

### Basic Usage
```python
from sparse_coding import decompose
import torch

# Define a target representation and dictionary
target = torch.randn(512)  # e.g., an embedding to be decomposed
dictionary = [torch.randn(512) for _ in range(10)]  # list of dictionary atoms where each atom is an embedding

# Decompose target on the dictionary
coefficients = decompose(
    target=target,
    dl_dict=dictionary,
    tau=0.95,      
    alpha=0.05,    
    normalize=True 
)
```

### Examples
We provide two demonstration cases in `test_decomposition.py`:

1. **Simple Diagonal Matrix Decomposition**: Decomposes a diagonal matrix into its basis vectors to validate the process on a controlled toy problem.

2. **CLIP Embedding Decomposition**: Decomposes CLIP embeddings. For example, decomposing the phrase `"a very cute dog"` on a collection of words like `["dog", "pet", "cute", "small", "cat", ...]` identifies the most important contributing concepts.

Run the examples with:
```bash
python test_decomposition.py
```