# ⚡ LeetGPU: High-Performance CUDA Kernels

![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Jetson%20Orin%20AGX-black?style=for-the-badge&logo=linux&logoColor=white)

A collection of optimized CUDA kernels solving challenges from [LeetGPU](https://leetgpu.com).
Focusing on **memory coalescing**, **shared memory tiling**, and **warp primitives** for embedded edge devices (Jetson
Orin).

## 📂 Engineering Log

| #   | Problem Name          | Date Solved  |              Solution              | Key Concepts                            |
|:----|:----------------------|:-------------|:----------------------------------:|:----------------------------------------|
| 001 | Vector Addition       | Feb 06, 2025 |  [Code](./src/001_vector_add.cu)   | Global Memory, Grid-Stride Loop         |
| 002 | Matrix Multiplication | Feb 06, 2025 |  [Code](./src/002_matrix_mul.cu)   | 2D Indexing, Memory Coalescing          |
| 003 | Color Inversion       | Feb 07, 2025 | [Code](./src/003_color_invert.cu)  | `uchar4` Types, Casts                   |
| 004 | Matrix Transpose      | Feb 08, 2025 |   [Code](./src/004_transpose.cu)   | Shared Memory (Bank Conflict Avoidance) |
| 005 | Matrix Add            | Feb 08, 2025 |    [Code](./src/005_mat_add.cu)    | Add matrix                              |
| 006 | Convolution 1D        | Feb 10, 2025 |  [Code](./src/006_convolution.cu)  | for loop inside kernel                  |
| 007 | Reverse Array         | Feb 10, 2025 | [Code](./src/007_reverse_array.cu) | swap inside kernel                      |
| 008 | Reverse Array         | Feb 11, 2025 |     [Code](./src/008_relu.cu)      | if inside kernel                        |

## 🛠️ Build & Run

This project uses **CMake** with a custom automation script for rapid iteration.

### **1. Create a New Solution**

Generate a new kernel file with boilerplate code:

```bash
./create.sh reduction_sum
# Creates src/006_reduction_sum.cu automatically