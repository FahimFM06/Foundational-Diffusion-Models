import streamlit as st
import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. DYNAMIC SHAPE-SHIFTING U-NET
# ==========================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.transform = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t):
        h = self.conv1(x)
        h = h + self.time_mlp(t)[:, :, None, None]
        return self.transform(h) + self.residual_conv(x)

class SharedUNet(nn.Module):
    # Added dynamic channel variables so the app adapts to the specific checkpoint
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=256, up1_in=384, up2_in=192):
        super().__init__()
        self.up1_in = up1_in
        self.up2_in = up2_in
        
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = Block(64, 128, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = Block(128, 256, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid1 = Block(256, 256, time_emb_dim)
        self.mid2 = Block(256, 256, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_block1 = Block(up1_in, 128, time_emb_dim) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_block2 = Block(up2_in, 64, time_emb_dim) 
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, t):
        t = self.time_mlp(t)
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        p2 = self.pool1(x2)
        x3 = self.down2(p2, t)
        p3 = self.pool2(x3)
        mid = self.mid1(p3, t)
        mid = self.mid2(mid, t)
        
        u1 = self.up1(mid)
        # Dynamically route the skip connection based on how the model was trained
        if self.up1_in == 384:
            u1 = torch.cat([u1, x3], dim=1)
        else:
            u1 = torch.cat([u1, p2], dim=1)
        u1 = self.up_block1(u1, t)
        
        u2 = self.up2(u1)
        # Dynamically route the second skip connection (Fixes the DDPM vs SDE mismatch)
        if self.up2_in == 192:
            u2 = torch.cat([u2, x2], dim=1) 
        else:
            u2 = torch.cat([u2, x1], dim=1) 
        u2 = self.up_block2(u2, t)
        return self.outc(u2)

# ==========================================
# 2. SCHEDULER & AUTO-DETECT LOADER
# ==========================================
class DDIMScheduler:
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cpu"):
        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def set_timesteps(self, num_inference_steps=50):
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        self.timesteps = timesteps.flip(0).to(self.device)

    def step(self, predicted_noise, timestep, prev_timestep, current_image):
        alpha_cumprod_t = self.alphas_cumprod[timestep]
        alpha_cumprod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0, device=self.device)
        pred_original_sample = (current_image - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev) * predicted_noise
        return torch.sqrt(alpha_cumprod_t_prev) * pred_original_sample + dir_xt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model(model_path):
    try:
        # Load the raw dictionary of weights
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(state_dict, nn.Module):
            state_dict = state_dict.state_dict()
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # AUTO-DETECT SHAPES: Peek at the layers inside the file
        up1_in_channels = clean_state_dict['up_block1.conv1.weight'].shape[1] if 'up_block1.conv1.weight' in clean_state_dict else 384
        up2_in_channels = clean_state_dict['up_block2.conv1.weight'].shape[1] if 'up_block2.conv1.weight' in clean_state_dict else 192
        
        # Build the model matching the exact dimensions found inside the file
        model = SharedUNet(up1_in=up1_in_channels, up2_in=up2_in_channels).to(device)
        model.load_state_dict(clean_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load `{model_path}`. Error details: {e}")
        return None

# ==========================================
# 3. STREAMLIT UI WITH TABS
# ==========================================
st.set_page_config(page_title="Diffusion Deblur", layout="wide")
st.title("Foundational Diffusion Models")

tab1, tab2 = st.tabs(["📖 Project Description", "🛠️ Deblurring Studio"])

with tab1:
    st.header("Project Overview")
    st.write("""
    This application is a demonstration of a project built from scratch to explore **four foundational mathematical architectures** for Generative AI. 
    The objective is to show how diffusion models can be used to process and clarify heavily degraded or blurry images.
    """)
    
    st.subheader("The Four Models Available")
    st.markdown("""
    * **DDPM:** The standard probabilistic framework.
    * **DDIM:** A deterministic process allowing for faster generation.
    * **Improved DDPM:** Utilizes a cosine noise schedule for better structural preservation.
    * **Score SDE:** A continuous-time framework utilizing stochastic differential equations.
    """)
    
    st.subheader("How to Use")
    st.write("1. Navigate to the **Deblurring Studio** tab.")
    st.write("2. Select your preferred diffusion model from the dropdown.")
    st.write("3. Upload your target image.")
    st.write("4. Click the 'Run Pipeline' button to view the result.")

with tab2:
    st.header("Deblurring Studio")
    
    model_choice = st.selectbox(
        "Step 1: Select Diffusion Model Architecture", 
        ["DDIM (Fastest)", "DDPM", "Improved DDPM", "Score SDE"]
    )
    
    weight_files = {
        "DDIM (Fastest)": "ddpm_unet_cifar10.pth",
        "DDPM": "ddpm_unet_cifar10.pth",
        "Improved DDPM": "improved_ddpm_unet_cifar10.pth",
        "Score SDE": "score_sde_unet_cifar10.pth"
    }

    model = load_model(weight_files[model_choice])

    if model is None:
        st.warning("Awaiting valid model weights to proceed.")
        st.stop()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Step 2: Upload Image")
        uploaded_file = st.file_uploader("Choose an image to clarify...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Step 3: Result")
        if uploaded_file is not None:
            if st.button(f"Run {model_choice} Pipeline"):
                with st.spinner('Running Reverse Diffusion Process...'):
                    
                    transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                    
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    scheduler = DDIMScheduler(device=device)
                    scheduler.set_timesteps(50)
                    
                    current_image = input_tensor
                    
                    for i, t in enumerate(scheduler.timesteps):
                        timestep_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                        prev_t = scheduler.timesteps[i + 1] if i < len(scheduler.timesteps) - 1 else -1
                        
                        with torch.no_grad():
                            predicted_noise = model(current_image, timestep_tensor)
                            
                        current_image = scheduler.step(predicted_noise, t, prev_t, current_image)
                    
                    final_image = (current_image.clamp(-1, 1) + 1) / 2.0
                    final_image = final_image[0].cpu().permute(1, 2, 0).numpy()
                    
                    st.image(final_image, caption=f"Processed via {model_choice}", use_column_width=True)
                    
                    st.info("Note: The model was trained on CIFAR-10, meaning the final output is processed at a 32x32 resolution.")
                    st.success("Generation Complete!")
