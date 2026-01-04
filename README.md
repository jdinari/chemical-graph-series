# 🧪 Chemical Graph Series

A progressive educational journey from basic **cheminformatics** to state-of-the-art **Graph Neural Networks (GNNs)** and **Molecular Transformers**. This series covers everything from representing molecules as graphs to predicting chemical properties using advanced deep learning architectures.

![Molecular Graph Representation](molGraph.png)

---

## 🎯 Who Is This For?

This series is designed for:

- **Computational chemists** looking to apply deep learning to molecular data
- **ML engineers** interested in graph neural networks with a chemistry application
- **Drug discovery researchers** wanting to build property prediction models
- **Students** with basic Python and chemistry knowledge

**Prerequisites**: Basic Python (loops, functions, data structures) and fundamental chemistry (molecular structure, bonds, functional groups). No prior experience with RDKit, graph theory, or deep learning required—we teach everything from scratch.

---

## 🚀 Curriculum Overview

The course is structured into 7 sequential notebooks, progressively building from foundations to production-ready models.

| Lesson | Title | Key Concepts | Time |
| :--- | :--- | :--- | :---: |
| **01** | [Building Graphs](./notebooks/01_Building_Graphs.ipynb) | SMILES parsing, RDKit, Mol-to-Graph, Feature extraction | 45-60 min |
| **02** | [Positional Encoding](./notebooks/02_Positional_Encoding.ipynb) | Laplacian Eigenvectors, RWPE, Spectral Analysis | 60-75 min |
| **03** | [GAT Model](./notebooks/03_GAT_Model.ipynb) | Graph Attention Networks, Message Passing, Multi-head Attention | 75-90 min |
| **04** | [Sparse Attention](./notebooks/04_Sparse%20Attention.ipynb) | Efficiency in Graph Transformers, Virtual Edges, Locality | 60-75 min |
| **05** | [Full Graph Transformer](./notebooks/05_Full_Graph_Transformer.ipynb) | Global Self-Attention, Edge Features, Deep Architectures | 90-105 min |
| **06** | [Advanced Graph Models](./notebooks/06_Advanced_Graph_Models.ipynb) | GraphGPS, E(3)-GNNs, Equivariance, Hybrid Architectures | 90-105 min |
| **07** | [Modelling & Predictions](./notebooks/07_Modelling_and_Predictions.ipynb) | Property Prediction (ESOL, FreeSolv), Training Pipelines | 120-150 min |

**Total Estimated Time**: ~9-11 hours

---

## 📚 Learning Path

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FOUNDATIONS (Lessons 01-02)                      │
│  • Molecular representations    • Feature extraction                   │
│  • Graph structures             • Positional encodings                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION MECHANISMS (Lessons 03-04)               │
│  • Local attention (GAT)        • Sparse patterns                      │
│  • Message passing              • Scalability                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED ARCHITECTURES (Lessons 05-06)               │
│  • Graph Transformers           • GraphGPS                             │
│  • Global context               • Equivariant networks                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION (Lesson 07)                         │
│  • Real datasets (ESOL, FreeSolv)    • Model comparison                │
│  • Training pipelines                • Deployment                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Setup & Installation

This project uses `pyproject.toml` for dependency management. It is recommended to use [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

### Using `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/ChemicalGraphSeries.git
cd ChemicalGraphSeries

# Sync environment and install all dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook
```

### Using `pip`

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install rdkit torch torch-geometric networkx matplotlib pandas jupyter py3dmol scipy

# Launch Jupyter
jupyter notebook
```

### Verify Installation

```python
# Run this in a notebook cell to verify everything works
from rdkit import Chem
import torch
import torch_geometric
import networkx as nx

print(f"RDKit: {Chem.rdBase.rdkitVersion}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch Geometric: {torch_geometric.__version__}")
print("✅ All dependencies installed successfully!")
```

---

## 📂 Project Structure

```
ChemicalGraphSeries/
├── notebooks/
│   ├── 01_Building_Graphs.ipynb      # Foundations: SMILES, RDKit, graphs
│   ├── 02_Positional_Encoding.ipynb  # Spectral graph theory & RWPE
│   ├── 03_GAT_Model.ipynb            # Graph Attention Networks
│   ├── 04_Sparse Attention.ipynb     # Efficient attention patterns
│   ├── 05_Full_Graph_Transformer.ipynb  # Complete transformer architecture
│   ├── 06_Advanced_Graph_Models.ipynb   # GraphGPS, E(3)-GNNs
│   └── 07_Modelling_and_Predictions.ipynb  # Real-world applications
├── molGraph.png                      # Visual for documentation
├── pyproject.toml                    # Project dependencies
├── uv.lock                           # Locked dependency versions
├── main.py                           # Utility scripts
└── README.md                         # This file
```

---

## 🧪 Requirements

| Requirement | Version |
|-------------|---------|
| **Python** | ≥ 3.13 |
| **RDKit** | latest |
| **PyTorch** | latest |
| **PyTorch Geometric** | latest |
| **NetworkX** | latest |
| **matplotlib** | latest |
| **pandas** | latest |
| **py3Dmol** | ≥ 2.5.3 |
| **scipy** | ≥ 1.16.3 |

---

## 🎓 What You'll Build

By the end of this series, you will have:

1. **Molecular featurization pipelines** — Convert any SMILES string into ML-ready graph representations
2. **Custom GNN architectures** — GATs, Graph Transformers, and hybrid models
3. **Property prediction models** — Trained on ESOL (solubility) and FreeSolv (solvation energy) benchmarks
4. **Interpretable AI** — Visualize attention weights to understand what your model "sees"
5. **Production-ready code** — Deployable models for real-world molecular property prediction

---

## 📖 Key Topics Covered

### Cheminformatics
- SMILES and SMARTS notation
- Molecular visualization (2D, 3D, conformer ensembles)
- Substructure matching and pharmacophore identification

### Graph Theory
- Molecules as graphs (atoms = nodes, bonds = edges)
- Adjacency and Laplacian matrices
- Spectral graph theory and eigenvector decomposition

### Deep Learning
- Message passing neural networks
- Attention mechanisms (single-head, multi-head, sparse)
- Transformer architectures adapted for graphs
- Equivariant neural networks (E(3)-GNNs)

### Practical ML
- Feature engineering for molecular properties
- Train/validation/test splitting with scaffold awareness
- Hyperparameter tuning and cross-validation
- Model interpretation and error analysis

---

## 🔗 Resources & Further Reading

**RDKit Documentation**: https://www.rdkit.org/docs/  
**PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/  
**DeepChem**: https://deepchem.io/  
**OGB Molecular Benchmarks**: https://ogb.stanford.edu/

**Key Papers**:
- Veličković et al. (2018) — Graph Attention Networks
- Rampášek et al. (2022) — GraphGPS
- Dwivedi et al. (2021) — Benchmarking GNNs

---

## 📝 License

This project is for educational purposes. Feel free to use, modify, and share with attribution.

---

<p align="center">
  <strong>Ready to start?</strong> Open <a href="./notebooks/01_Building_Graphs.ipynb">Lesson 01: Building Graphs</a> and begin your journey!
</p>
