## ResFCN256-3D-Reconstruction

ResFCN256-3D-Reconstruction implements a custom deep learning pipeline for dense 3D facial geometry estimation. At its core is a ResFCN256 (Residual Fully Convolutional Network) trained for 100 epochs on a dataset of 63,000 augmented images to perform direct regression of 3D facial coordinates from single 2D inputs. The model outputs a 3D Position Map, a dense matrix where RGB pixel values encode high-fidelity (X,Y,Z) spatial data rather than color intensity. To interpret this raw tensor output, the repository includes a specialized visualization engine that decodes the abstract position map into a verifiable 3D point cloud, demonstrating the model's ability to capture subtle organic curvature and depth via direct neural regression.

## Tools Included

### 1. Neural Mesh Inspector (`src/app/2D-3D_Visualization.py`)
An interactive Streamlit tool to visualize the raw neural network output.
* **Input:** Single portrait image.
* **Output:** Side-by-side comparison of input photo, raw position map (the "blob"), and rotatable 3D point cloud.
* **Feature:** Real-time background masking and point density adjustments.

### 2. 3D Model Generator (`src/app/toply.py`)
A production-ready utility to convert 2D images into standard 3D files.
* **Function:** Exports the inferred geometry as a **.PLY** (Polygon File Format) file.
* **Compatibility:** Generated files can be opened in Blender, MeshLab, or Windows 3D Viewer.

## Model Architecture

* **Type:** ResFCN256 (Encoder-Decoder with Residual Blocks) 

[Image of Encoder-Decoder neural network architecture]

* **Input Resolution:** 256x256x3
* **Training Data:** 63,000 samples (CelebA dataset processed via MediaPipe)
* **Loss Function:** MSE Loss (Mean Squared Error)

## Tech Stack
* **PyTorch:** Deep Learning framework for custom training loop and inference.
* **OpenCV:** Image preprocessing and matrix transformations.
* **Streamlit:** Interactive UI for real-time model interaction.

## Folder Structure
```

/ ResFCN256-3D-Reconstruction
│
├── data/
│   ├── input/              # Place your test images here (e.g., test.jpg)
│   └── output/             # Generated results (blobs, PLY files, images)
│
├── models/
│   └── prnet_trained.pth   # The trained Neural Network weights (~54MB)
│
├── src/app
│       ├── toply.py     # Streamlit App: Converts 2D Photo -> 3D Model
│       └── 2D-3D_Visualization.py   # Streamlit App: Visualizes the raw Neural Output
│  
│
└── requirements.txt        # Python dependencies

```

## Disclaimer
**Pretrained Weights**

This repository uses **Git LFS** to store large files.

To clone the repository *with* pretrained weights , run:

```bash
git lfs install
git clone https://github.com/luqspy/ResFCN256-3D-Reconstruction.git
```

## How to Run
**Create and activate environment**

```
conda create -n loomis_env python=3.9
conda activate loomis_env

```

**Install Requirements**
```
pip install -r requirements.txt

```

**Run Neural Visualizer**
```
streamlit run src/app/2D-3D_Visualization.py

```

**Run the 2D-3D Generator App**
```
streamlit run src/app/toply.py

```


