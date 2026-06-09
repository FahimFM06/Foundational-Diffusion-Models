# Foundational Diffusion Models: Image Deblurring Studio

An interactive web application and end-to-end mathematical pipeline demonstrating the evolution of generative diffusion models. Built entirely from scratch using PyTorch and deployed via Streamlit.

## 📖 About the Project
This project explores the core mathematical architectures that power modern Generative AI. Instead of relying on pre-built high-level APIs, this repository contains raw, scratch-built implementations of four distinct diffusion frameworks. 

The application takes heavily degraded, blurry, or noisy images and computationally reverses the degradation process to reconstruct the original structures, utilizing a shared dynamic U-Net backbone trained on the CIFAR-10 dataset.

## 🧠 The Four Architectures

1. **DDPM (Denoising Diffusion Probabilistic Models):** The standard baseline. Uses a discrete Markov chain of 1,000 timesteps with a linear noise schedule to slowly destroy and reconstruct data.
2. **DDIM (Denoising Diffusion Implicit Models):** The speed champion. Re-derives the reverse process as non-Markovian and deterministic, allowing the model to skip timesteps and generate images up to 20x faster without retraining.
3. **Improved DDPM:** The structural champion. Implements a cosine-based noise schedule to prevent the image from losing structural information too early in the forward process, alongside stabilized variance predictions to prevent gradient explosions.
4. **Score SDE (Stochastic Differential Equations):** The continuous-time foundation. Transitions from discrete timesteps to continuous time $t \in (0, 1]$, using a Variance Preserving SDE and solving the reverse generation process numerically via the Euler-Maruyama method.

## 🛠️ Technical Highlights
* **Dynamic Shape-Shifting U-Net:** The neural network automatically detects channel dimension mismatches (e.g., 128 vs 192 channel skip connections) inside `.pth` weight files and dynamically rebuilds its architecture at runtime to prevent shape collision errors.
* **Streamlit UI Pipeline:** Fully integrated frontend that resizes custom user uploads to the required $32 \times 32$ tensor format, mathematically applies the correct forward-degradation noise, and visualizes the reverse denoising loop in real-time.
* **Bulletproof Inference:** Built-in safeguards against numerical instability, including epsilon injections and gradient clipping.

## 🚀 Installation & Setup

To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/foundational-diffusion-models.git](https://github.com/yourusername/foundational-diffusion-models.git)
   cd foundational-diffusion-models
