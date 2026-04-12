# DA2-CVAE

## 📖 Overview
**DA2-CVAE** is a novel framework based on **conditional variational autoencoders (CVAE)**, integrated with dimensionality reduction, clustering, and data balancing of sampling data to consturcut the protein free energy landscape effectively.

The workflow involves:
1.  **Preprocessing**: Enhanced sampling (not included), PCA dimensionality reduction,  K-Means clustering, and data balancing.
2.  **CVAE Training**: Training a CVAE conditioned on cluster labels to learn the latent distribution of protein structures.
3.  **CVAE Sampling**: Generating new conformations, relabelling them, and calculating weights.
4.  **Reweighting**: Calculating critical collective variables (CVs, not included) and using weights to reconstruct Free Energy Landscape (FEL).

## 🛠️ Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install torch numpy matplotlib scikit-learn joblib mdtraj prody
```
*   **PyTorch**: For building and training the CVAE model.
*   **MDTraj / ProDy**: For handling protein PDB files and coordinate transformations.
*   **Scikit-learn**: For PCA, K-Means, and MinMaxScaler.
*   **Matplotlib**: For visualization of loss curves and free energy surfaces.

## 📂 Project Structure

```text
DA2-CVAE/
├── 2RVD/*              # Example protein system: Chignolin, PDB ID: 2RVD
├── cvae.py              # Main entry point: orchestrates training, reconstruction, and reweighting
├── preprocess.py        # Data preprocessing: XYZ extraction, PCA, and K-Means clustering
├── dataset.py           # PyTorch Dataset class: handles data loading, scaling, and cluster balancing
├── model.py             # CVAE Model definition: Encoder, Decoder, and BatchNorm layers
├── loss.py              # Loss functions: BCE/MSE Reconstruction Loss + KL Divergence
├── cvae_train.py        # Training loop for the CVAE
├── cvae_recon.py        # Sampling/Reconstruction module: samples from latent space
├── reweight.py          # Reweighting logic: projects generated data back to PCA/KMeans space
├── get_trainset_cv.py   # Utility: extracts collective variables (CVs) for specific clusters
├── tools.py             # Helper functions for loading data and plotting utilities
├── plot.py              # Visualization scripts: FES, RMSD histograms, convergence plots
└── README.md
```

## 🚀 Quick Start

### 1. Configuration
Edit `cvae.py`, `preprocess.py`, and `plot.py` to set your specific task parameters:
```python
task = '2RVD'           # Name of the task
task_affix = None       # Affix for task-specific files
beta = 1                # Weight for KL divergence in CVAE loss
n_clusters = 4          # Number of K-Means clusters
N_confomation_per_cluster = 10000 # Samples per cluster for training
batch_size = 40         # Batch size for training
hidden_size = [256, 128, 64, 16] # Hidden layer sizes for Encoder/Decoder
latent_size = 4         # Dimension of the latent space
num_epochs = 100        # Training epochs
loss = bce_loss         # Loss function
num_samples = 50000     # Number of samples in recon
recon_affix = 'randU'   # Affix for recon
pdbfile = '2RVD.pdb'    # PDB file
```

### 2. Preprocessing (Optional if data exists)
If raw MD trajectories are available, run `preprocess.py` to generate PCA components (70%) and K-Means labels:
```bash
python preprocess.py
```
This generates:
*   `{task}/`: Task directory.
*   `{task}/coor_xyz_{task}.npy`: Flattened coordinates.
*   `{task}/pca_{task}.pkl` & [.npy]: PCA model and reduced data.
*   `{task}/kmeans_{task}_n{N}.pkl` & [.npy]: K-Means model and cluster labels.
*   `{task}/kmeans_inertia.dat`: Inertia values for K-Means models.
We have provided example data files for `2RVD` in `2RVD/`. 

### 3. Run Full Pipeline
Execute the main script to train, generate, and reweight:
```bash
python cvae.py
```
**Workflow executed by `cvae.py`:**
1.  **Train CVAE**: Trains the model using `cvae_train.py`. Saves encoder/decoder weights and loss logs.
2.  **Reconstruct/Generate**: Uses `cvae_recon.py` to sample `num_samples` new conformations from the latent space.
3.  **Relabel**: Uses `reweight.py` to project generated structures onto the original PCA space and assign K-Means cluster labels.
4.  **Extract CVs**: Uses `get_trainset_cv.py` to save Collective Variables (e.g., RMSD, HBonds) for each cluster.
This generates:
*   `{task}/models/`: Directory for model weights.
*   `{task}/loss/`: Directory for model loss.
*   `{task}/recon/recon_.../`: Directory for generated structures.
*   `{task}/data/`: Directory for CVs data.
*   `{task}/cluster_n{N}_i_{N_confomation_per_cluster}.npy`: selected structures' indexes for training in i-th cluster.
*   `{task}/models/[encoder/decoder]_....pth`: Model weights for ecoder/decoder, saved every 10 epoch.
*   `{task}/loss/[kl/recon/total]_....txt`: KL, Reconstruction, and Total loss.
*   `{task}/recon/recon_.../protein_*.pdb`: Generated PDB files.
*   `{task}/recon/recon_.../coor_xyz_..._recon.npy`: Flattened coordinates for generated structures.
*   `{task}/pca_..._recon.npy`: PCs for generated structures.
*   `{task}/kmeans_..._recon.npy`: K-Means labels for generated structures.
*   `{task}/kdist_....npy`: Distance to nearest cluster for training structures.
*   `{task}/kdist_..._recon.npy`: Distance to nearest cluster for generated structures.
*   `{task}/data/c?_..._cv.out`: CVs data for each cluster.

### 4. FEL Construction and Visualization
Run [plot.py] to generate analysis figures and save in `{task}/fig`:
```bash
python plot.py
```
**Generated Plots:**
*   `kmeans_raw_...`: Original data colored by cluster.
*   `kmeans_train_...`: Trainset data overlaid on original density.
*   `kmeans_recon_...`: Generated data overlaid on original density.
*   `{cv}_weighted_...`: 1D FEL.
*   `{cv}_2D_raw_...`: Trainset 2D FEL.
*   `{cv}_2D_weighted_...`: Reweighted 2D FE.
*   `loss_...`: Training loss convergence curves.

## 📝 Output Files
*   `{task}/models/`: Saved PyTorch state dicts for Encoder and Decoder.
*   `{task}/loss/`: Text files containing epoch-wise loss values.
*   `{task}/recon/`: Generated PDB files and concatenated XYZ arrays.
*   `{task}/data/`: Extracted Collective Variables (RMSD, HBonds) for each cluster.
*   `{task}/fig/`: All generated matplotlib figures.
