# Foundational Diffusion Models: Image Deblurring Studio

[![Live Application](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit)](https://foundational-diffusion-models-u85qe4w6zsc6g9oc96xouv.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/FahimFM06/Foundational-Diffusion-Models/tree/main)

An interactive web application and end-to-end mathematical pipeline demonstrating the evolution of generative diffusion models. Built entirely from scratch using PyTorch and deployed via Streamlit.

This project takes heavily degraded, blurry, or noisy images and computationally reverses the degradation process to reconstruct the original structures, utilizing a shared dynamic U-Net backbone trained on the CIFAR-10 dataset.

## 🚀 Live Demo
**Try the interactive web application here:** [Deblurring Studio on Streamlit](https://foundational-diffusion-models-u85qe4w6zsc6g9oc96xouv.streamlit.app/)

---

## 🧠 Comparative Analysis of the Four Architectures
This project implements four distinct mathematical frameworks to solve the same reverse-diffusion problem. Here is an analysis of how they function, along with their distinct advantages and trade-offs.

### 1. DDPM (Denoising Diffusion Probabilistic Models)
* **How it works:** The standard baseline. It uses a discrete Markov chain of 1,000 timesteps with a linear noise schedule to slowly destroy and reconstruct data, adding a small amount of probabilistic Gaussian noise at each backward step.
* **Why it works well:** Highly stable to train and provides a strong mathematical foundation for understanding score-matching.
* **Where it struggles:** Extremely slow inference. It *must* evaluate all 1,000 steps sequentially to generate an image, making it computationally expensive for real-time applications.

### 2. DDIM (Denoising Diffusion Implicit Models)
* **How it works:** Re-derives the reverse process as non-Markovian and deterministic. It removes the random noise injection ($\sigma = 0$) during the reverse loop.
* **Why it works well:** **The Speed Champion.** Because the process is deterministic, DDIM can safely "skip" timesteps. It can generate images of comparable quality to DDPM in just 50 steps instead of 1,000, resulting in a massive ~20x speedup.
* **Where it struggles:** Deterministic sampling can sometimes lead to slightly less diverse outputs compared to purely probabilistic models.

### 3. Improved DDPM
* **How it works:** Upgrades the standard DDPM by replacing the linear noise schedule with a **Cosine Noise Schedule** and implementing learned/stabilized variance predictions.
* **Why it works well:** **The Quality Champion (Discrete Time).** The linear schedule destroys too much structural information too early. The cosine schedule degrades the image smoothly, allowing the neural network to preserve structural edges and vibrant colors much more effectively.
* **Where it struggles:** It retains the slow, 1,000-step sequential inference bottleneck of standard DDPM and requires careful gradient clipping during training to prevent numerical instability (exploding gradients).

### 4. Score SDE (Stochastic Differential Equations)
* **How it works:** Transitions from discrete timesteps to continuous time $t \in (0, 1]$. It uses a Variance Preserving SDE (VP-SDE) to map the data distribution to noise, and solves the reverse generation process numerically via the Euler-Maruyama method.
* **Why it works well:** **The Mathematical Foundation.** By treating time as continuous, it generalizes all discrete diffusion models into a single framework. It handles complex, high-dimensional continuous data distributions exceptionally well.
* **Where it struggles:** It is the most mathematically complex to implement and computationally heavy to solve, as numerical differential solvers require fine time-slicing.

---

## 🛠️ Step-by-Step Implementation Journey
Building these architectures from scratch required navigating several complex engineering hurdles. Here is how the pipeline was constructed:

### Step 1: The Dynamic Shape-Shifting U-Net
The core backbone of the project is a customized U-Net equipped with Sinusoidal Positional Embeddings to track the timestep $t$. 
* **The Challenge:** During training experiments across the four models, the architecture naturally shifted (e.g., skip connections generating 128-channel vs. 192-channel concatenations). 
* **The Solution:** The `SharedUNet` class was engineered to be dynamic. At runtime, the application peeks inside the `.pth` checkpoint file, auto-detects the channel dimensions the specific model was trained with, and rebuilds its internal architecture to perfectly catch the incoming weights, entirely preventing tensor shape collisions.

### Step 2: The Forward Process (Mathematical Degradation)
To successfully run an Image-to-Image pipeline on the Streamlit frontend, feeding a clean image directly to the reverse loop causes massive numerical instability (rendering purely white images). 
* **The Implementation:** The app mathematically degrades the user's uploaded image using the specific variance schedule ($\bar{\alpha}_t$) of the chosen model, jumping directly to a noisy state before passing it to the neural network.

### Step 3: The Reverse Schedulers & Inference
Instead of relying on high-level APIs like HuggingFace `diffusers`, the mathematical schedulers were written from scratch in PyTorch. 
* Epsilon injections (`1e-8`) and rigorous tensor clamping (`[-1.0, 1.0]`) were implemented across the discrete DDPM/DDIM formulas and the continuous Euler-Maruyama solver to prevent division-by-zero errors and black-square mode collapse.

### Step 4: Streamlit Integration & Processing
The models were trained on the CIFAR-10 dataset, which dictates a strict $32 \times 32$ pixel resolution. The frontend utilizes `torchvision.transforms` to compress custom user uploads into this format, run the complex tensor math in the background, and seamlessly decode the output back into viewable RGB images.

---

## 🚀 Local Installation

To run this project locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/FahimFM06/Foundational-Diffusion-Models.git](https://github.com/FahimFM06/Foundational-Diffusion-Models.git)
   cd Foundational-Diffusion-Models
