# 🖼️ Image Super-Resolution using GANs (SRGAN)

This project implements a complete **Super-Resolution GAN (SRGAN)** pipeline to generate high-resolution images (×4) from low-resolution inputs.  
The framework includes a **ResNet-based generator**, **convolutional discriminator**, and a **perceptual loss** module using pretrained VGG-19 features.

The project is fully implemented in **PyTorch**, trained on the **DIV2K benchmark dataset**, and evaluated using both distortion metrics (PSNR/SSIM) and perceptual metrics (LPIPS).  
Visual comparisons demonstrate clear improvements over traditional bicubic interpolation.

---

## 📌 Project Objective

The goal is to perform **perceptual image super-resolution**, reconstructing high-frequency textures and sharper edges beyond what pixel-based models (SRCNN, bicubic) can achieve.

Key aims:

- Upscale input images by **4×** using GAN-based generative modeling  
- Preserve semantic content using **VGG-based perceptual loss**  
- Improve texture realism and perceptual fidelity  
- Evaluate and compare against bicubic interpolation  
- Save training artifacts for reproducibility (checkpoints, visual samples)

---

## 📁 Dataset Details — DIV2K

- **Dataset:** DIV2K (standard SR benchmark)  
- **Images:** 1000 high-resolution natural images  
- **Paired Data:**  
  - HR image (ground truth)  
  - Corresponding LR ×4 bicubic downsampled image  
- **Patch Extraction:**  
  - HR patches: 96×96  
  - LR patches: 24×24  
- **Why DIV2K?**  
  - High diversity  
  - High resolution  
  - Used widely in SRGAN, ESRGAN, Real-ESRGAN research  

All LR–HR pairs were normalized to `[0, 1]` and prepared using PyTorch DataLoaders.

---

## 🔧 Preprocessing Workflow

- Bicubic downsampling (×4)  
- Bicubic upsampling to match HR size (input to generator)  
- Random patch extraction  
- Tensor normalization  
- Mini-batch loading on GPU  
- Patch-based augmentation (random crops)

This strategy reduces computation and improves texture diversity.

---

## 🧱 Baseline: SRCNN (Sanity Check)

Before SRGAN training:

- A simple **SRCNN model** was trained briefly  
- Confirms dataset correctness, LR–HR alignment  
- Produces high-PSNR but smooth images  
- Serves as a distortion-focused baseline  

---

## 🧠 SRGAN Architecture

### **Generator (ResNet with PixelShuffle)**
- 9×9 initial convolution  
- Deep residual blocks with PReLU  
- Two PixelShuffle ×2 upsampling blocks  
- Final 9×9 reconstruction layer  
- Outputs HR RGB image  

### **Discriminator**
- PatchGAN-style CNN classifier  
- Strided convolutions  
- LeakyReLU activations  
- Fully connected binary classifier  

### **Perceptual Feature Extractor**
- Pretrained **VGG-19**  
- Feature maps from early layers used to compute **content/perceptual loss**  

---

## 🎯 Loss Functions

- **Pixel Loss (L1/MSE)**  
  Ensures color consistency and structural alignment

- **Content Loss (VGG Feature Loss)**  
  Encourages semantic similarity and texture correctness

- **Adversarial Loss**  
  Trains generator to fool discriminator and produce natural-looking textures

- **Final Generator Loss:**  

Loss balancing ensures stable convergence and prevents artifacts.

---

## ⚙️ Training Procedure

### **Phase 1 — Warm-Up (2 epochs)**
- Only pixel + content loss  
- Stabilizes generator  
- Avoids early texture hallucination  
- Ensures color fidelity  

### **Phase 2 — Adversarial Training (16 epochs)**
- Generator + discriminator trained alternately  
- Discriminator updated every 6 steps  
- Pixel loss retained for stability  
- Output values clamped to `[0, 1]`

### **Training Setup**
- Batch size: 16  
- HR patch: 96×96  
- LR patch: 24×24  
- Optimizer: Adam (LR = 1e-4)  
- Total epochs: 18  
- All artifacts saved (generator checkpoints, PSNR/SSIM/LPIPS)

---

## 📊 Quantitative Results

| Metric | Bicubic | SRGAN (Ours) |
|--------|---------|---------------|
| **PSNR** | 26.16 dB | 17.15 dB |
| **SSIM** | 0.772 | 0.270–0.295 |
| **LPIPS ↓** | 0.372 | **0.265** |

### Interpretation

- Bicubic is better in *pixel similarity* (PSNR/SSIM)  
- **SRGAN is far superior in perceptual realism**  
- Lower LPIPS → more human-perceived similarity  
- Matches results from SRGAN, ESRGAN literature  

---

## 🌄 Qualitative Results

### **4-Panel Comparison**
- LR → Bicubic → SRGAN → HR  
- SRGAN outputs:  
- Sharper edges  
- Richer textures  
- Higher contrast  
- Realistic fine detail  
- No checkerboard artifacts  

### **Zoomed Patches**
- Bicubic: smooth, blurry  
- SRGAN: detailed textures (bricks, foliage, surfaces)  
- HR: ground truth  

Visual comparisons strongly show perceptual gains.

---

## ⭐ Key Findings

- SRGAN excels at **perceptual SR**, not distortion-based SR  
- Best metric: **LPIPS 0.265** (much better than bicubic)  
- Stable training via warm-up + controlled discriminator updates  
- Generator effectively reconstructs high-frequency structure  
- Produces visually realistic outputs with preserved color balance  

---

## 🚀 Future Improvements

- Train for 50–100 epochs for stronger texture learning  
- Upgrade to **ESRGAN** or **Real-ESRGAN**  
- Explore:  
- Style loss  
- Histogram loss  
- Multi-scale SR  
- Diffusion-assisted refinement  
- Integrate Real-World Degradation modeling  
- Progressive and multi-scale GANs  

---

## 📦 Summary Table

| Component | Description |
|----------|-------------|
| Model | SRGAN (GAN-based perceptual SR) |
| Dataset | DIV2K |
| Scale Factor | ×4 |
| Generator | ResNet + PixelShuffle |
| Discriminator | CNN PatchGAN |
| Best Metric | LPIPS = 0.265 |
| Visual Outcome | Sharper, realistic textures |
| Training | Warm-up + adversarial fine-tuning |
| Framework | PyTorch |

---

## 🧱 Tech Stack

- **PyTorch**  
- **Torchvision**  
- **LPIPS**  
- **NumPy, Pillow**  
- **Matplotlib**  
- **Google Colab (GPU)**  
