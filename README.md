# MNIST in CUDA: My Implementation Journey 🚀

Learning CUDA optimization through MNIST digit recognition: My step-by-step implementation journey

---

## About This Project
I'm working through this project to learn how to optimize neural network code from high-level PyTorch down to optimized CUDA. This repository documents my personal implementation journey as I build a simple MLP for MNIST classification with progressively lower-level code.

The original tutorial this is based on provides a fantastic learning path that I'm following to understand the performance implications of different implementation approaches.

**My GitHub**: https://github.com/JJKSweaty/MNIST

---

## Model Architecture

| Component | Details |
|----------|---------|
| Network | 2-layer MLP (784 → 256 → 10) |
| Dataset | MNIST (10,000 training samples) |
| Training | Batch size 8, 10 epochs |
| Activation | ReLU |
| Loss | Cross-entropy |
| Optimizer | SGD (lr = 0.01) |

---

## Implementation Versions

I’ll be implementing each of these versions to understand performance differences:

### ✅ v1.py — PyTorch Baseline
- High-level PyTorch operations with GPU acceleration  
- Performance reference implementation  

### ✅ v2.py — NumPy CPU Implementation
- Pure CPU-based forward/backward  
- Full understanding of neural network math  

### ⚙️ v3.c — C/CPU Implementation
- Manual memory management  
- Timing instrumentation  

### 🔄 v4.cu — Naive CUDA Kernels
- First CUDA acceleration  
- GPU memory transfer + kernel launch learning  

### 🔜 v5.cu — cuBLAS Optimized
- Optimized GEMM with SGEMM  
- Better memory efficiency  
- Goal: exceed PyTorch timing  

---

## CUDA Setup & Requirements

Requires: **NVIDIA GPU (Compute Capability ≥ 5.0)**

```bash
# Check CUDA installation
nvcc --version

# Compile CUDA versions
nvcc -o v4 v4.cu
nvcc -o v5 v5.cu -lcublas
'''

## What I'm Learning

Through this project, I'm gaining hands-on experience with:

- Neural network implementation from scratch
- CUDA programming fundamentals
- Memory management optimization techniques
- Performance profiling and bottleneck identification
- Trade-offs between high-level frameworks and low-level code

## Performance Expectations

I expect to see dramatic performance differences between implementations:

| Version | Implementation | Expected Performance |
|---------|----------------|---------------------|
| v1.py | PyTorch CUDA | Baseline (fast) |
| v2.py | NumPy CPU | Much slower than baseline |
| v3.c | C CPU | Faster than NumPy but still slow |
| v4.cu | Naive CUDA | Better, but not optimal |
| v5.cu | cuBLAS CUDA | Should exceed PyTorch performance |

## Progress Tracking

I'll update this README as I complete each implementation with my personal observations and learning points.

## Key CUDA Concepts I'm Learning

- Row vs Column Major matrix layouts
- Tensor Core acceleration
- Efficient memory access patterns
- Kernel launch optimization
- Advanced techniques like unified memory and CUDA streams

## Acknowledgements

This project is based on a tutorial that provides a great learning path for CUDA optimization. I'm implementing it myself to gain hands-on experience with these concepts.