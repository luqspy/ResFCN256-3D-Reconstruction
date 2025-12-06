#importing stuff we need
import streamlit as st
import torch
import cv2
import numpy as np
import sys
import os

#fix path so we can find the model file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.training.model import ResFCN256

#setup the tab info
st.set_page_config(page_title="3D Face Generator", page_icon="🗿", layout="centered")
MODEL_PATH = "models/prnet_trained.pth"

#function to write the ply file string
def create_ply_string(vertices, colors):
    #header stuff for ply format
    header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    body = []
    #colors need to be 0-255 int
    colors_int = (colors * 255).astype(np.uint8)
    
    for v, c in zip(vertices, colors_int):
        #flipping Y so it's not upside down, and X to mirror it
        body.append(f"{-v[0]:.4f} {-v[1]:.4f} {v[2]:.4f} {c[0]} {c[1]} {c[2]}")
    
    return header + "\n".join(body)

#load the brain, cache it so it's fast
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #gpu check
    if not os.path.exists(MODEL_PATH): return None, device
    
    model = ResFCN256().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device

#UI stuff
st.title("🗿 3D Face Generator")
st.markdown("Convert any portrait into a **Downloadable 3D Model (.ply)**.")

model, device = load_model()
if model is None:
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

#upload pic
uploaded_file = st.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])

if uploaded_file:
    #read the file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True, caption="Original")

    with col2:
        with st.spinner("Generating 3D Geometry..."):
            #resize because model only knows 256x256
            img_resized = cv2.resize(image, (256, 256))
            input_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
            
            #run the ai!
            with torch.no_grad():
                output = model(input_tensor)
            pos_map = output.cpu().squeeze().permute(1, 2, 0).numpy()
            
            #aggressive masking to clean up noise
            mask = pos_map[:, :, 2] > 0.02
            
            #get the valid points
            valid_points = pos_map[mask]
            colors_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
            valid_colors = colors_rgb[mask]
            
            #make the file
            ply_data = create_ply_string(valid_points, valid_colors)
            
            st.success("✅ Model Ready!")
            #download button logic
            st.download_button(
                label="⬇️ Download .PLY Model",
                data=ply_data,
                file_name="generated_face.ply",
                mime="text/plain"
            )
            st.info(f"Contains {len(valid_points)} vertices.")

else:
    st.info("Upload an image to start.")