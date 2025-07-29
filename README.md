# uTDSP


### Transformer-based Diffusion and Spectral Priors Model For Hyperspectral Pansharpening  
📄 [[Paper Link (IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 2025)]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11085108)

### Hyperspectral Pansharpening with Transformer-based Spectral Diffusion Priors  
📄 [[Paper Link (The IEEE/CVF Winter Conference on Applications of Computer Vision 2025)]](https://openaccess.thecvf.com/content/WACV2025W/GeoCV/papers/Jiang_Hyperspectral_Pansharpening_with_Transformer-based_Spectral_Diffusion_Priors_WACVW_2025_paper.pdf)


**Authors:**  
[Hongcheng Jiang](https://jianghongcheng.github.io/)  
[Zhiqiang Chen](https://sse.umkc.edu/profiles/zhiqiang-chen.html)

---
---

## 🔍 Overview

The goal is to reconstruct a **high-resolution hyperspectral image (HR-HSI)** by fusing a **low-resolution hyperspectral image (LR-HSI)** and a **high-resolution panchromatic image (HR-PCI)**. Unlike conventional methods that rely on paired HR-HSI ground truth, **uTDSP is entirely unsupervised**, leveraging **spectral priors** and a **transformer-based diffusion model** to guide the reconstruction process.


---

## 🧠 Key Features

- 🎯 **Unsupervised Learning**: Learns directly from LR-HSI and HR-PCI without requiring any ground-truth HR-HSI.
- 🌀 **Spectral Diffusion Process**: Incorporates a transformer-based denoiser within a diffusion framework.
- 🧩 **Spectral Prior Integration**: Enforces spectral consistency using priors extracted from the LR-HSI.
- ⚖️ **Adaptive Loss Balancing**: Combines spectral fidelity loss and diffusion consistency for robust reconstruction.
- 🏆 **SOTA Results**: Achieves superior performance across multiple benchmark datasets.

---


## 📈 Performance Gains

**uTDSP** achieves consistent PSNR improvements across diverse airborne and satellite datasets, outperforming both supervised and unsupervised baselines.

### ✅ Airborne Datasets

- **Chikusei**  
  - uTDSP: **26.86 dB**  
  - Best prior method (DDLPS*): 26.85 dB  
  - 🔺 **+0.01 dB (+0.04%)**

- **Indian Pines**  
  - uTDSP: **25.79 dB**  
  - Best prior method (DIP-HyperKite): 25.15 dB  
  - 🔺 **+0.64 dB (+2.54%)**

- **PaviaC**  
  - uTDSP: **28.53 dB**  
  - Best prior method (DDLPS*): 27.52 dB  
  - 🔺 **+1.01 dB (+3.67%)**

- **PaviaU**  
  - uTDSP: **30.68 dB**  
  - Best prior method (GPPNN): 29.86 dB  
  - 🔺 **+0.82 dB (+2.75%)**

### ✅ Satellite Datasets

- **Botswana**  
  - uTDSP: **31.61 dB**  
  - Best prior method (DIP-HyperKite): 30.24 dB  
  - 🔺 **+1.37 dB (+4.53%)**

- **ZY1-02D**  
  - uTDSP: **31.22 dB**  
  - Best prior method (SFIM*): 28.23 dB  
  - 🔺 **+2.99 dB (+10.59%)**

---

### 📌 Summary

uTDSP consistently improves PSNR by:
- **+0.01–1.01 dB (up to +3.7%)** on airborne datasets  
- **+1.37–2.99 dB (up to +10.6%)** on satellite datasets  

These results confirm uTDSP’s strong generalization and superior reconstruction quality across sensing platforms—all achieved without supervision.



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

### 📷  Satellite Datasets
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

@ARTICLE{11085108,
  author={Jiang, Hongcheng and Chen, ZhiQiang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Transformer-based Diffusion and Spectral Priors Model For Hyperspectral Pansharpening}, 
  year={2025},
  volume={},
  number={},
  pages={1-17},
  keywords={Hyperspectral imaging;Pansharpening;Diffusion models;Transformers;Estimation;Bayes methods;Noise reduction;Image reconstruction;Earth;Degradation;Hyperspectral imaging;pansharpening;spectral priors;diffusion model;transformer;remote sensing},
  doi={10.1109/JSTARS.2025.3590685}}

@inproceedings{jiang2025hyperspectral,
  title={Hyperspectral Pansharpening with Transformer-Based Spectral Diffusion Priors},
  author={Jiang, Hongcheng and Chen, ZhiQiang},
  booktitle={Proceedings of the Winter Conference on Applications of Computer Vision},
  pages={581--590},
  year={2025}
}
```


## 📬 Contact

If you have any questions, feedback, or collaboration ideas, feel free to reach out:

- 💻 Website: [jianghongcheng.github.io](https://jianghongcheng.github.io/)
- 📧 Email: [hjq44@mail.umkc.edu](mailto:hjq44@mail.umkc.edu)
- 🏫 Affiliation: University of Missouri–Kansas City (UMKC)


