# Lipid Nanoparticle TEM Image Detection

This repository demonstrates an end-to-end machine learning pipeline for the detection and classification of Lipid Nanoparticles (LNPs) in Transmission Electron Microscopy (TEM) micrographs. By leveraging **High-Performance Computing (HPC)** and **Digital Twin** simulation, this project addresses the scarcity of labeled biological datasets.

## Key Technical Competencies

* **HPC Architecture:** Large-scale data orchestration and model training on cluster environments using SLURM.
* **Multi-GPU Optimization:** Implementation of Distributed Data Parallel (DDP) training across 4x NVIDIA L40S GPUs.
* **Synthetic Data Engineering:** Development of a derivatized digital twin to simulate complex negative stain TEM physics.
* **Object Detection:** Advanced fine-tuning of YOLO11n architectures for high-density particle environments.

---

## Technical Approach

### 1. Digital Twin Simulation (`generateimages_wbckgrd.py`)

To overcome the lack of centralized LNP datasets, a custom simulation engine was developed to generate high-fidelity synthetic micrographs.

* **Morphology Generation:** Algorithms to simulate Solid LNPs, Unilamellar/Multilamellar Vesicles, and Multivesicular Liposomes.
* **TEM Physics Simulation:** Uses **Elastic Transformations** and Gaussian noise modeling to replicate lipid membrane deformations and the "staining gradient" characteristic of negative stain TEM.
* **Background Hardening:** A dedicated generation task for pure background noise, significantly reducing false-positive rates in dense environments.

### 2. Scalable Training Pipeline (`Train_Model_V4.py`)

The training workflow is optimized for speed and reproducibility:

* **Hardware:** Optimized for 4x NVIDIA L40S GPUs (184GB total VRAM).
* **Data Staging:** Automated pipeline that cleans, organizes, and generates YOLO-standard YAML configurations on the fly.
* **Distributed Training:** Utilizes 16 CPU workers and AMP (Automatic Mixed Precision) to maximize throughput.

---

## Training Results

The model achieved exceptional performance metrics on the synthetic test set, demonstrating the effectiveness of the digital twin in training high-precision detectors.

| Class | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| **Global Average** | 2040 | 59988 | **1.0** | **1.0** | **0.995** | **0.869** |
| Solid LNP | 1020 | 35275 | 1.0 | 1.0 | 0.995 | 0.992 |
| Vesicular Structures | 1020 | 24713 | 1.0 | 1.0 | 0.995 | 0.832 |

### Domain Adaptation & Future Work

While the model shows perfect convergence on synthetic data, current research is focused on bridging the **Domain Gap**. Real-world TEM images exhibit extreme diversity due to localized lab preparation methods and staining inconsistencies. The high synthetic accuracy provides a robust "pre-trained" foundation; future iterations will utilize **Transfer Learning** on smaller, curated sets of real-world data to enhance generalization across varied laboratory environments.

---

## Repository Structure

* `generateimages_wbckgrd.py`: Digital twin generation engine.
* `Train_Model_V4.py`: Multi-GPU training orchestration script.
* `generateimages_wbckgrd.slurm`: HPC batch configuration for data generation.
* `Train_Model_V4.slurm`: HPC batch configuration for 4-GPU training nodes.

---

## Getting Started

### Installation

```bash
pip install ultralytics opencv-python numpy scipy

```

### Usage

1. **Generate Dataset:** `sbatch generateimages_wbckgrd.slurm`
2. **Execute Training:** `sbatch Train_Model_V4.slurm`
