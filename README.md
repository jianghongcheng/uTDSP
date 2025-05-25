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
  <img src="https://github.com/jianghongcheng/FW-SAT/blob/main/Figures/Network.png" width="800"/>
</p>


---

## 🧩 Flexible Window Attention Module

<p align="center">
  <img src="https://github.com/jianghongcheng/FW-SAT/blob/main/Figures/Flexible_Window_Att.png" width="800"/>
</p>


---

## 🖼️ Visual Results

<p align="center"><strong>Comparison with State-of-the-Art Methods</strong></p>
<p align="center">
  <img src="https://github.com/jianghongcheng/FW-SAT/blob/main/Figures/Visual_Result.png" width="800"/>
</p>


---




## 📊 Quantitative Results

<p align="center"><b>Table: Quantitative comparison with state-of-the-art methods on the validation dataset (PSNR/SSIM)</b></p>

<div align="center">

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th colspan="2">×8</th>
      <th colspan="2">×16</th>
    </tr>
    <tr>
      <th>PSNR</th>
      <th>SSIM</th>
      <th>PSNR</th>
      <th>SSIM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>EDSR</td>
      <td>25.66</td>
      <td>0.8394</td>
      <td>22.59</td>
      <td>0.7562</td>
    </tr>
    <tr>
      <td>SwinIR</td>
      <td>24.98</td>
      <td>0.8170</td>
      <td>21.22</td>
      <td>0.7277</td>
    </tr>
    <tr>
      <td>HAN</td>
      <td>25.86</td>
      <td>0.8430</td>
      <td>22.69</td>
      <td>0.7591</td>
    </tr>
    <tr>
      <td>GRL</td>
      <td>25.59</td>
      <td>0.8405</td>
      <td>22.38</td>
      <td>0.7500</td>
    </tr>
    <tr>
      <td><b>FW-SAT (Ours)</b></td>
      <td><b>27.80</b></td>
      <td><b>0.8815</b></td>
      <td><b>24.61</b></td>
      <td><b>0.8116</b></td>
    </tr>
  </tbody>
</table>

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


