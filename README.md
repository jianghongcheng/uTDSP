# uTDSP

### Hyperspectral Pansharpening with Transformer-based Spectral Diffusion Priors  
📄 [[Paper Link (WACV 2025 Workshop)]](https://openaccess.thecvf.com/content/WACV2025W/GeoCV/papers/Jiang_Hyperspectral_Pansharpening_with_Transformer-based_Spectral_Diffusion_Priors_WACVW_2025_paper.pdf)


**Authors:**  
[Hongcheng Jiang](https://jianghongcheng.github.io/)  
[Zhiqiang Chen](https://sse.umkc.edu/profiles/zhiqiang-chen.html)

---
---

## 🔍 Overview

**uTDSP** (Unsupervised Transformer-based Diffusion and Spectral Priors) is a novel framework for hyperspectral pansharpening.

The goal is to reconstruct a high-resolution hyperspectral image (HR-HSI) by fusing a low-resolution hyperspectral image (LR-HSI) and a high-resolution panchromatic image (HR-PCI). Unlike conventional methods that require paired HR-HSI ground truth, **uTDSP is entirely unsupervised**, leveraging **spectral priors** and **transformer-based diffusion** to guide the reconstruction process.

---

## 🧠 Key Features

- 🎯 **Unsupervised Learning**: Learns directly from LR-HSI and HR-PCI without requiring any ground-truth HR-HSI.
- 🌀 **Spectral Diffusion Process**: Incorporates a transformer-based denoiser within a diffusion framework.
- 🧩 **Spectral Prior Integration**: Enforces spectral consistency using priors extracted from the LR-HSI.
- ⚖️ **Adaptive Loss Balancing**: Combines spectral fidelity loss and diffusion consistency for robust reconstruction.
- 🏆 **SOTA Results**: Achieves superior performance across multiple benchmark datasets.

---



## 📈 Performance Gains

We propose **FW-SAT**, a Flexible Window-based Self-attention Transformer for thermal image super-resolution.

✅ **Results on Validation Set**

**×8 Upscaling:**
- FW-SAT achieves **27.80 dB / 0.8815** in PSNR/SSIM, outperforming all competitors:
  - **+2.82 dB / +0.0645** (**+11.29% PSNR / +7.89% SSIM**) vs. SwinIR
  - **+1.94 dB / +0.0385** (**+7.50% PSNR / +4.57% SSIM**) vs. HAN
  - **+2.21 dB / +0.0410** (**+8.64% PSNR / +4.88% SSIM**) vs. GRL
  - **+2.14 dB / +0.0421** (**+8.34% PSNR / +5.02% SSIM**) vs. EDSR

**×16 Upscaling:**
- FW-SAT achieves **24.61 dB / 0.8116**, again setting a new benchmark:
  - **+3.39 dB / +0.0839** (**+15.97% PSNR / +11.53% SSIM**) vs. SwinIR
  - **+1.92 dB / +0.0525** (**+8.46% PSNR / +6.91% SSIM**) vs. HAN
  - **+2.23 dB / +0.0616** (**+9.96% PSNR / +8.21% SSIM**) vs. GRL
  - **+2.02 dB / +0.0554** (**+8.94% PSNR / +7.32% SSIM**) vs. EDSR

These consistent improvements across scales and metrics validate FW-SAT’s strong generalization and superior spatial-spectral learning capabilities.


## 🔄 Diffusion Process Illustration

<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/diffusion.gif" width="800"/>
</p>

---

## 🧠 Network Architecture

<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/utdsp.png" width="800"/>
</p>


## 📊 Band-wise PSNR Comparison

The following plots illustrate the PSNR values for each spectral band across six benchmark datasets, highlighting the spectral fidelity of uTDSP compared to existing methods.

### 🛰️ Botswana
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/Botswana_PSNR.png" width="800"/>
</p>

### 🛰️ Chikusei
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/Chikusei_PSNR.png" width="800"/>
</p>

### 🛰️ Pavia Center
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/PaviaC_PSNR.png" width="800"/>
</p>

### 🛰️ Pavia University
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/PaviaU_PSNR.png" width="800"/>
</p>

### 🛰️ Indian Pines
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/indian_PSNR.png" width="800"/>
</p>

### 🛰️ Ziyuan-1
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/Ziyuan_PSNR.png" width="800"/>
</p>



---

## 🖼️ Visual Results

### 📷  Airborne datasets
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/result1.png" width="800"/>
</p>

### 📷  Satellite DatasetS
<p align="center">
  <img src="https://github.com/jianghongcheng/uTDSP/blob/main/Figures/result2.png" width="800"/>
</p>


---

## 📊 Quantitative Results on Airborne Datasets

*Metrics: PSNR ↑ (higher is better), SAM ↓, ERGAS ↓  
(* denotes an unsupervised method)*

---

<div align="center">

### 🛰️ Chikusei

| Method          | PSNR  | SAM    | ERGAS |
|-----------------|:-----:|:------:|:------:|
| DBDENet         | 25.02 | 6.5243 | 4.2316 |
| DHP-DARN        | 25.24 | 6.0044 | 3.9208 |
| DIP-HyperKite   | 25.63 | 5.4180 | 3.7059 |
| DMLD-Net        | 25.28 | 6.9856 | 4.1170 |
| GPPNN           | 25.17 | 6.5704 | 4.1423 |
| HyperPNN        | 25.34 | 5.7174 | 3.8096 |
| DDLPS*          | 26.85 | 5.3557 | 3.7616 |
| GSA*            | 24.21 | 6.2670 | 5.3903 |
| Indusion*       | 22.29 | 5.4171 | 5.0367 |
| PLRDiff*        | 26.18 | 6.3831 | 3.5699 |
| SFIM*           | 25.54 | **5.4171** | 4.0868 |
| **uTDSP***      | **26.86** | 5.9152 | **3.3178** |

</div>

---

<div align="center">

### 🛰️ Indian Pines

| Method          | PSNR  | SAM    | ERGAS |
|-----------------|:-----:|:------:|:------:|
| DBDENet         | 23.66 | **3.4962** | 1.6389 |
| DHP-DARN        | 23.97 | 3.6511 | 2.1060 |
| DIP-HyperKite   | 25.15 | 3.6619 | 1.4592 |
| DMLD-Net        | 24.03 | 3.9927 | 1.7721 |
| GPPNN           | 24.80 | 4.4834 | 1.5553 |
| HyperPNN        | 24.60 | 3.8811 | 1.5532 |
| DDLPS*          | 17.72 | 4.5282 | 10.0461 |
| GSA*            | 20.44 | 4.0885 | 5.0318 |
| Indusion*       | 4.83  | 3.8504 | 14.8366 |
| PLRDiff*        | 8.10  | 10.6969 | 19.8362 |
| SFIM*           | 24.42 | 3.8504 | 1.5483 |
| **uTDSP***      | **25.79** | 3.5621 | **1.2763** |

</div>

---

<div align="center">

### 🛰️ PaviaC

| Method          | PSNR  | SAM    | ERGAS |
|-----------------|:-----:|:------:|:------:|
| DBDENet         | 23.63 | 18.5981 | 6.7944 |
| DHP-DARN        | 26.70 | 12.4018 | 4.7917 |
| DIP-HyperKite   | 26.48 | 12.5846 | 4.9198 |
| DMLD-Net        | 26.03 | 16.8061 | 5.1832 |
| GPPNN           | 27.37 | 11.2643 | 4.4916 |
| HyperPNN        | 26.10 | 17.5919 | 5.1762 |
| DDLPS*          | 27.52 | **10.1478** | 4.3781 |
| GSA*            | 25.30 | 10.4678 | 5.6518 |
| Indusion*       | 25.84 | 10.4645 | 5.8683 |
| PLRDiff*        | 27.39 | 11.6904 | 4.6245 |
| SFIM*           | 24.99 | 10.4488 | 5.9011 |
| **uTDSP***      | **28.53** | 10.4176 | **4.2664** |

</div>

---

<div align="center">

### 🛰️ PaviaU

| Method          | PSNR  | SAM    | ERGAS |
|-----------------|:-----:|:------:|:------:|
| DBDENet         | 28.84 | 6.5032 | 2.7593 |
| DHP-DARN        | 29.27 | 6.8826 | 2.5350 |
| DIP-HyperKite   | 29.30 | 6.0972 | 2.5114 |
| DMLD-Net        | 28.66 | 6.8624 | 2.7985 |
| GPPNN           | 29.86 | **6.0788** | 2.4812 |
| HyperPNN        | 28.96 | 6.7555 | 2.6472 |
| DDLPS*          | 27.81 | 7.1405 | 4.3781 |
| GSA*            | 26.47 | 7.2522 | 2.9323 |
| Indusion*       | 25.82 | 7.8229 | 4.1539 |
| PLRDiff*        | 28.57 | 7.5217 | 2.9453 |
| SFIM*           | 25.66 | 7.8229 | 3.8125 |
| **uTDSP***      | **30.68** | 6.8019 | **2.4660** |

</div>

## 🛰️ Quantitative Results on Satellite Datasets

*Metrics: PSNR ↑ (higher is better), SAM ↓, ERGAS ↓  
(* denotes an unsupervised method)*

---

<div align="center">

### 🛰️ Botswana

| Method         | PSNR   | SAM     | ERGAS   |
|----------------|:------:|:-------:|:-------:|
| DBDENet        | 22.84  | 8.5207  | 11.3979 |
| DHP-DARN       | 28.85  | 4.9084  | 2.8164  |
| DIP-HyperKite  | 30.24  | 4.8305  | 2.1305  |
| DMLD-Net       | 26.87  | 6.5379  | 3.7552  |
| GPPNN          | 26.44  | 8.6439  | 3.8965  |
| HyperPNN       | 29.83  | 4.9803  | 2.2254  |
| DDLPS*         | 22.27  | 6.9539  | 17.5198 |
| GSA*           | 23.80  | 6.2035  | 11.6626 |
| Indusion*      | 15.30  | 5.4225  | 9.7633  |
| PLRDiff*       | 17.84  | 15.1475 | 9.0164  |
| SFIM*          | 26.81  | 5.4225  | 2.7995  |
| **uTDSP***     | **31.61** | **3.7777** | **1.9155** |

</div>

---

<div align="center">

### 🛰️ ZY1-02D

| Method         | PSNR   | SAM     | ERGAS   |
|----------------|:------:|:-------:|:-------:|
| DBDENet        | 11.39  | 22.3445 | 29.8753 |
| DHP-DARN       | 14.71  | 7.9514  | 24.0358 |
| DIP-HyperKite  | 19.73  | 2.1563  | 3.8904  |
| DMLD-Net       | 13.41  | 13.4045 | 11.3059 |
| GPPNN          | 14.35  | 12.1717 | 10.6018 |
| HyperPNN       | 19.42  | 2.7872  | 5.5515  |
| DDLPS*         | 26.03  | 1.8212  | 5.0919  |
| GSA*           | 16.60  | 4.5868  | 20.0515 |
| Indusion*      | 14.53  | **1.7358** | 6.7524  |
| PLRDiff*       | 26.77  | 2.5636  | 2.7032  |
| SFIM*          | 28.23  | **1.7358** | **1.6604** |
| **uTDSP***     | **31.22** | 1.7466  | 1.6917  |

</div>




## 📚 Citation

If you find this work helpful in your research, please cite:

```bibtex
@InProceedings{Jiang_2024_CVPR,
    author    = {Jiang, Hongcheng and Chen, Zhiqiang},
    title     = {Flexible Window-based Self-attention Transformer in Thermal Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3076--3085}
}
```


## 📬 Contact

If you have any questions, feedback, or collaboration ideas, feel free to reach out:

- 💻 Website: [jianghongcheng.github.io](https://jianghongcheng.github.io/)
- 📧 Email: [hjq44@mail.umkc.edu](mailto:hjq44@mail.umkc.edu)
- 🏫 Affiliation: University of Missouri–Kansas City (UMKC)


