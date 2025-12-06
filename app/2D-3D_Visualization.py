import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#import ResFCN256 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.training.model import ResFCN256

#configureing the brwoser tab 
st.set_page_config(page_title="Neural Mesh Inspector", page_icon="🧠", layout="wide")
MODEL_PATH = "models/prnet_trained.pth" #adding path to trained weights

#add performance boost to streamlit. Basically load model fucntion is run once, result are rmemembered
@st.cache_resource
def load_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #check for cude cores
    if not os.path.exists(MODEL_PATH): #check model path
        return None, device
    
    model = ResFCN256().to(device) #create architectire of NN
    state_dict = torch.load(MODEL_PATH, map_location=device) #read weights
    model.load_state_dict(state_dict) #inject weights into model
    model.eval() #switches model to eval mode
    return model, device #returen model adn device to main

st.title("🧠 Neural Mesh Inspector") #title

model, device = load_model() #load model
if model is None:
    st.error(f"Model not found at {MODEL_PATH}")
    st.stop()

#defining sidebar
st.sidebar.header("Data Controls") 
#allow user to adjust the mask to see all the points
mask_threshold = st.sidebar.slider("Background Cutoff (Z-Depth)", 0.000, 0.100, 0.010, step=0.001, format="%.3f")
#checkbox to force show everything even noise
show_all = st.sidebar.checkbox("Show Raw Output (No Mask)", value=False)

st.sidebar.header("Visual Controls")

view_mode = st.sidebar.radio("View Mode", [
    "2D Position Map (The Blob)", 
    "3D Point Cloud (Textured)", 
    "3D Point Cloud (Geometry Only)"
])
st.sidebar.markdown("---")

#conditional block for sliders to change viewing angle. 
if "3D Point Cloud" in view_mode:
    elev = st.sidebar.slider("Vertical Rotation", -90, 90, 0)
    azim = st.sidebar.slider("Horizontal Rotation", -180, 180, -90)
    stride = st.sidebar.slider("Point Density", 1, 10, 2)
    point_size = st.sidebar.slider("Point Size", 0.1, 5.0, 1.5)

#take input
uploaded_file = st.file_uploader("Upload Portrait", type=["jpg", "png", "jpeg"])

if uploaded_file:
    #read uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    img_resized = cv2.resize(image, (256, 256)) #recize if necessary
    
    #preparing image
    input_tensor = torch.from_numpy(img_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device) #pixel to float
    with torch.no_grad(): #specify non-training cycle
        output = model(input_tensor) #calculate outpou
    pos_map = output.cpu().squeeze().permute(1, 2, 0).numpy()

    #logic to handle the hole in the middle
    if show_all:
        mask = np.ones((256, 256), dtype=bool) #select everything
    else:
        mask = pos_map[:, :, 2] > mask_threshold #user-controlled cutoff
    
    valid_points = pos_map[mask]
    
    #2d visualization blob
    if view_mode == "2D Position Map (The Blob)":
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Photo")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
        with col2:
            st.subheader("Neural Output (Position Map)")
            #Normalize to 0-255 for display
            blob_vis = cv2.normalize(pos_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(cv2.cvtColor(blob_vis, cv2.COLOR_BGR2RGB), use_column_width=True, caption="R=X, G=Y, B=Z")
            st.info("This blurry square contains the exact 3D coordinates for every pixel.")

    #3d visualization
    elif "3D Point Cloud" in view_mode:
        #subsample
        pts = valid_points[::stride]
        
        if len(pts) == 0:
            st.warning("No points! Try lowering the Cutoff slider.")
        else:
            # Determine Color
            if view_mode == "3D Point Cloud (Textured)":
                colors = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
                valid_colors = colors[mask]
                cols = valid_colors[::stride]
            else:
                #geometry only
                cols = 'c' # 'c' is cyan
            
            #plot
            fig = plt.figure(figsize=(10, 10))
            fig.patch.set_alpha(0.0)
            
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_facecolor('none')
            
            #plot Points - flipping Y so it's upright
            ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=point_size, alpha=0.6)
            
            #cleanup
            ax.axis('off') 
            ax.grid(False)
            ax.view_init(elev=elev, azim=azim)
            
            st.pyplot(fig)
            st.markdown(f"**Rendering {len(pts)} points.**")

else:
    st.info("Upload an image to inspect the brain.")