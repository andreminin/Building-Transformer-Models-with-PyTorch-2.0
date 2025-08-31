# Preface

This repository is a **fork of a PyTorch learning project for beginners**, with significant modifications:

- The original project was tailored for **Apple Silicon with Metal Performance Shaders (MPS)**.   
  This fork instead focuses on **Linux/CUDA**, tested on **Ubuntu 24.04** with CUDA 12.9 and **NVIDIA GPU (16GB VRAM)**.  
- Includes **fixes for imports**, compatibility updates, and code adjustments for **current versions of PyTorch and dependencies**.
- Notebooks with the suffix **`_MOD`** in their names are **refactored versions** of the originals with code improvements and updates.  
- Serves as my **personal training project** for running and experimenting with transformer models in a CUDA environment.

# Installation & Requirements

A new `requirements.txt` file is included with the necessary dependencies.  
To set up the environment:

1. **Create a virtual environment** (you can specify Python version if needed):
   ```bash
   uv venv --seed
   # or explicitly with Python 3.10
   uv venv --seed -p 3.10
   ```

2. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Jupyter kernel**:
   
   ```bash
   python -m ipykernel install --user --name=building_transformers --display-name="Models-with-PyTorch"
   ```
5. **Set Huggingface services token environment variable**: 
   Register on Huggingface and set HF_TOKEN in ~/.bashrc 
   
   ```bash
   export HF_TOKEN=hf_your_hf_token
   #reload:
   source ~/.bashrc
   ```
6. **Run Jupyter Notebook**:  
   On a localhost, start the notebook, it will open project in browser:
   ```bash
   jupyter notebook
   ```
   On a remote machine, start the notebook server with:
   ```bash
   jupyter notebook --no-browser --ip=host
   ```
   If running on a host server, you can use:
   ```bash
   jupyter notebook --no-browser --ip=*
   ```
   or specify a particular IP address instead of `*`.

---

# Building Transformer Models with PyTorch 2.0

Your key to transformer based NLP, vision, speech, and multimodalities

This is the repository for [Building Transformer Models with PyTorch 2.0
](https://bpbonline.com/products/building-transformer-models-with-pytorch-2-0?_pos=1&_sid=1a44a1b38&_ss=r&variant=43297384005832),published by BPB Publications.

<img src="9789355517494.jpg">

## About the Book
This book covers transformer architecture for various applications including NLP, computer vision, speech processing, and predictive modeling with tabular data. It is a valuable resource for anyone looking to harness the power of transformer architecture in their machine learning projects.

The book provides a step-by-step guide to building transformer models from scratch and fine-tuning pre-trained open-source models. It explores foundational model architecture, including GPT, VIT, Whisper, TabTransformer, Stable Diffusion, and the core principles for solving various problems with transformers. The book also covers transfer learning, model training, and fine-tuning, and discusses how to utilize recent models from Hugging Face. Additionally, the book explores advanced topics such as model benchmarking, multimodal learning, reinforcement learning, and deploying and serving transformer models.

In conclusion, this book offers a comprehensive and thorough guide to transformer models and their various applications.

## What You Will Learn
• Understand the core architecture of various foundational models, including single and multimodalities.

• Step-by-step approach to developing transformer-based Machine Learning models.

• Utilize various open-source models to solve your business problems.

• Train and fine-tune various open-source models using PyTorch 2.0 and the Hugging Face ecosystem.

• Deploy and serve transformer models.

• Best practices and guidelines for building transformer-based models.
