# AgroVision AI 🌾🤖

An easy-to-use image classifier framework for crop disease detection. This repository contains data balancing, training pipelines, and an interactive inference interface built on top of PyTorch and torchvision using a fine-tuned ResNet-18 backbone combined with Gemini and Edge-TTS optimizations.

🔗 **Live Link:** https://agrovision-ai.streamlit.app

---

## What you'll find in this repo

- requirements.txt — Project dependencies and Python library specifications.
- app.py — Streamlit-based web application featuring futuristic glassmorphism UI, Test-Time Augmentation (TTA), and intelligent fallback structures.
- train.py — Core training pipeline supporting mixed precision (AMP), stratified validation splits, StepLR scheduling, and strong data augmentations.
- balance_dataset.py — Data pipeline script utilizing automated random lightweight augmentations to balance out minority dataset classes.
- utils.py — Utility scripts handling dataset scanning, directory mapping, and clean dictionary checkpoint loading.
- metrics.csv — Training history file logging accuracy and loss performance metrics across epochs.

---

## Quick start

### 1. Clone the Repository
git clone https://github.com/HarshTiwari1131/Agrovision.git
cd Agrovision

### 2. Set Up a Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

### 3. Verify PyTorch Environment (Optional)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

### 4. Run the Streamlit Application
streamlit run app.py

---

## Dataset layout

Place your custom training image files inside a folder named `Train/` at the root directory level. Organize your images strictly adhering to the standard ImageFolder layout structure (one distinct subdirectory per class label):
Train/
├─ Healthy Maize/
│   ├─ img001.jpg
│   └─ img002.jpg
├─ Maize Blight/
│   ├─ img001.jpg
│   └─ img002.jpg
└─ Tomato Rust/
├─ img001.jpg
└─ img002.jpg

Supported extension matching patterns: `.jpg`, `.jpeg`, `.png`, `.bmp` (configured inside `utils.py`).

---

## Training (Detailed CLI Workflow)

The main entrypoint for executing your training experiments is `train.py`.

### Example command execution:
python train.py --data-dir Train --epochs 15 --batch-size 32 --lr 1e-3 --output model.pth

### Key CLI arguments and parameters:
- `--data-dir` (str, default: 'Train') — Direct path pointing to your image subdirectories.
- `--epochs` (int, default: 10) — The maximum number of explicit training loops to process.
- `--batch-size` (int, default: 32) — Sample size processed per step forward pass.
- `--image-size` (int, default: 224) — Target resolution dimensions downscaled for ResNet ingestion.
- `--lr` (float, default: 1e-3) — Base structural learning rate for Adam optimizer execution.
- `--output` (str, default: 'model.pth') — Target save path for the resulting high-accuracy checkpoint weights file.
- `--val-split` (float, default: 0.2) — Dataset percentage isolated for evaluating target generalization accuracy.
- `--augment` (str, choices: ['basic', 'strong'], default: 'basic') — Select transform strength applied across training layers.
- `--weighted-sampler` (flag) — Overrides uniform picking by adjusting sample frequencies based on inverse class balances.
- `--balance-data` (flag) — Dynamically runs synthetic image generation routines over undersampled classes ahead of model setup loops.

---

## Inference / Dashboard Platform

Launch the dashboard UI locally by running:
streamlit run app.py

### Core architectural implementation details:
- **Test-Time Augmentation (TTA):** Feeds original images alongside flipped and varied rotational counterparts through the active model instance to produce averaged prediction probabilities.
- **Dynamic Diagnostics & Recovery:** Computes a confidence matching criteria matrix, presenting localized advice vectors alongside interactive dataset metrics layout cards.

---

## Troubleshooting

- **Folder Path Validation Warnings:** If your user interface throws a missing workspace folder flag, ensure that the target source location directory structure matches the required syntax conventions precisely.
- **CUDA Out Of Memory Errors:** Scale down the target operational `--batch-size` parameter block configuration step size or decrease model ingestion frame heights via the explicit `--image-size` variable parameter definitions.

---

## 🤝 How to Collaborate

Contributions, forks, and optimizations are highly welcome!
1. Fork the workspace repository: https://github.com/HarshTiwari1131/Agrovision/fork
2. Track developments inside separate functional workspaces: git checkout -b feature/AmazingFeature
3. Document commits, push updates online, and initialize code verification procedures by requesting a Pull Request tracking merger review.

---

**Developed with ❤️ by Harsh Tiwari (https://github.com/HarshTiwari1131)**