# GeoFeat: Multi-Order Geometric Feature Fusion for Robust Local Descriptor Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **GeoFeat** 是一个创新的局部特征描述子学习框架，通过融合多阶几何信息（深度、法向量、曲率）并结合 AFTAttention 机制，显著提升了图像匹配在复杂几何场景下的鲁棒性。

## 📅 项目动态 (Project Status)


## 🎯 核心功能 (Core Features)

### 1. 多阶几何特征融合 (Multi-Order Geometric Fusion)

GeoFeat 首次系统性地引入了 0-2 阶几何信息，弥补了传统纹理描述子在几何模糊区域的缺陷：

| 几何阶数 | 特征类型 | 物理意义 | 关键作用 |
|---------|---------|---------|---------|
| **0阶** | 深度 (Depth) | 3D 位置 | 提供绝对尺度感知 |
| **1阶** | 法向量 (Normal) | 表面朝向 | 捕捉平面与斜面变化 |
| **2阶** | 曲率 (Curvature) | 表面弯曲 | 识别角点、边缘与复杂曲面 (k₁, k₂, K, H) |

### 2. 几何感知注意力与增强 (Geometric Attention & Enhancement)

- **AFTAttention (Attention-Free Transformer)**: 
  采用高效的 AFT 机制替代传统 Self-Attention，在保持全局感受野的同时大幅降低计算复杂度，实现几何特征的全局交互。

- **GeometricLocalRefiner**:
  专为几何特征设计的局部细化模块。利用深度可分离卷积 (Depthwise Separable Conv) 注入局部归纳偏置，防止特征在几何不连续处（如物体边缘）发生过度平滑。

### 3. 高分辨率几何解耦 (High-Res Geometry Unfolding)

传统的下采样过程会丢失高频几何细节。GeoFeat 采用了 **Geometry Unfolding** 技术：
- 将高分辨率的几何图（如 8x8 块）展开并堆叠到通道维度。
- 使 1/8 分辨率的特征图能够感知原始分辨率的几何细节，从而在低分辨率下也能保持高精度的几何感知能力。

### 4. 混合骨干网络架构 (Hybrid Backbone Architecture)

- **多尺度融合**: 融合 $x3, x4, x5$ 层特征，兼顾局部纹理细节与高层语义信息。
- **灵活性**: 支持 Standard CNN 与 RepVGG 骨干网络，可根据推理速度需求灵活切换。

