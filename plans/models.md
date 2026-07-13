Ran command: `py -m venv venv`
Ran command: `venv\Scripts\activate`

Installing PyTorch with full CUDA support (like in your provided command) is notoriously large because it bundles all the required NVIDIA libraries (cuDNN, cuBLAS, etc.) directly into the package.

Here is a breakdown of what to expect in terms of storage and your lighter alternatives:

### 1. Size of PyTorch with CUDA (cu130 / cu121 / cu118)
If you run that exact command, the downloaded wheels and the extracted installation will take up roughly **2.5 GB to 4 GB** of disk space in your virtual environment. The exact size fluctuates slightly based on the specific CUDA version, but it is always massive.

### 2. Can you use a lighter/smaller version?
**Yes, absolutely.** The primary way to drastically reduce the size is to install the **CPU-only version** of PyTorch. 

If you are just running basic inference (like generating embeddings via HuggingFace models for a RAG pipeline) and you either don't have a dedicated NVIDIA GPU or don't strictly need real-time GPU acceleration, the CPU version is perfectly fine and significantly smaller. 

The CPU-only version takes up only around **200 MB to 250 MB** of disk space (roughly 10x smaller).

### How to install the lighter CPU-only version
Instead of pointing to the `cu130` index, you point `pip` to the `cpu` index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Summary Recommendation
- **Use your original command** if you have an NVIDIA GPU, your system has plenty of storage, and you need maximum performance for generating HuggingFace embeddings.
- **Use the CPU command** if you want to save gigabytes of space, are deploying to a standard cloud server without a GPU, or if CPU generation speeds are acceptable for your application.