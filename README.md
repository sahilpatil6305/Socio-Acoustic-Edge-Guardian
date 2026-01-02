# Socio-Acoustic Edge Guardian: Adaptive Deepfake Detection
### Team Cyber Guardian | eRaksha Hackathon 2026

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Edge%20%7C%20Mobile-green.svg)]()

**Problem Statement:** Agentic AI for Deepfake Detection & Authenticity Verification.

The **Socio-Acoustic Edge Guardian** acts as a **"Digital Bloodhound"** for tactical environments. Unlike standard detectors that just track the "scent" (acoustic cues), our agent:
1.  **Tracks the Scent**: Analyzes acoustic features via ResNet50/Wav2vec.
2.  **Checks the Trail**: Verifies if the *content* matches the *voice* using **Semantic Grounding**.
3.  **Remembers Every Scent**: Uses **RegO** to learn new deepfakes without forgetting old ones.
4.  **Tracks Through Storms**: Uses **F-SAT** to remain robust against field corruptions (MP3, Noise).

---

## Key Features
*   **The 'RegO' Learning Brain**: Partitions neurons (A, B, C, D) to enable **Continual Learning**. Updates "Fake" regions orthogonally to prevent catastrophic forgetting.
*   **Dual-Stream Perception**: Fuses **Acoustic** (Mel-spectrogram) and **Semantic** (Whisper-tiny) streams using **Optimal Transport (OT) Fusion**.
*   **Tactical Robustness (F-SAT)**: Specifically defends against attacks in the **4-8k Hz** range and handles **MP3/AAC** compression.
*   **Edge-Ready**: Uses a **5-layer SimpleMlp** backend for low-latency inference on CPU-only devices (Intel i7 / Smartphones).

## Repository Structure
*   `main.py`: **CLI Entry Point** for the agent.
*   `rego_fsat_classes.py`: **Core Logic** (RegO, F-SAT, OTFusion).
*   `optimize_model.py`: **Optimization Script** for Quantization & TorchScript.
*   `SYSTEM_ARCHITECTURE.md`: Detailed technical architecture and flowcharts.
*   `requirements.txt`: Project dependencies.

## Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/sahilpatil6305/Socio-Acoustic-Edge-Guardian.git
    cd socio-acoustic-guardian
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Optimize Model (Recommended)
Generate the edge-optimized model (`agent_optimized.pt`):
```bash
python optimize_model.py
```

### 2. Run Inference
Run the agent in detection mode:
```bash
python main.py --mode detect --input sample_audio.wav --verbose
```

## Performance
| Metric | Value | Note |
| :--- | :--- | :--- |
| **Accuracy** | **98.80%** | Outperforms RWM SOTA |
| **Inference Time** | **~124ms** | CPU-only (Optimized) |
| **Model Size** | **~35 MB** | Int8 Quantized |

## Documentation
*   [System Architecture](SYSTEM_ARCHITECTURE.md)

---
*Built for eRaksha Hackathon 2026 by Team Cyber Guardians.*
