import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

# conda install pytorch cpuonly -c pytorch

# Load lightweight MiDaS model for depth estimation
@st.cache_resource
def load_midas_model():
    model_type = "MiDaS_small"  # Lightweight model for low-end devices
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=True)
    midas.eval()
    device = torch.device("cpu")  # Use CPU for low-end devices
    midas.to(device)
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
    return midas, transform, device

# Function to process a single frame for depth map
def process_frame(frame, midas, transform, device):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    return depth.astype(np.uint8)

# Function to create a simple 3D effect (anaglyph)
def create_anaglyph(left_img, right_img):
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    anaglyph = np.zeros_like(left_img)
    anaglyph[:, :, 0] = left_img[:, :, 0]  # Red channel from left
    anaglyph[:, :, 1] = right_img[:, :, 1]  # Green channel from right
    anaglyph[:, :, 2] = right_img[:, :, 2]  # Blue channel from right
    return anaglyph

# Streamlit app
st.title("2D to 3D Video Converter")
st.write("Upload a short video to convert it to a 3D anaglyph effect. Works best with videos under 10 seconds on low-end devices.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
shift_pixels = st.slider("3D Depth Shift (pixels)", 1, 10, 5)

if uploaded_file is not None:
    # Save uploaded video temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Load model
    midas, transform, device = load_midas_model()

    # Read video
    cap = cv2.VideoCapture("temp_video.mp4")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video
    output_path = "output_3d.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    st.write("Processing video... This may take a while on low-end devices.")
    progress_bar = st.progress(0)

    # Process frames
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Get depth map
        depth = process_frame(frame, midas, transform, device)

        # Create left and right images for anaglyph
        left_img = frame.copy()
        right_img = frame.copy()
        for y in range(height):
            for x in range(width):
                shift = int(depth[y, x] / 255.0 * shift_pixels)
                if x + shift < width:
                    right_img[y, x] = frame[y, x + shift]
                else:
                    right_img[y, x] = 0

        # Create anaglyph image
        anaglyph = create_anaglyph(left_img, right_img)
        out.write(cv2.cvtColor(anaglyph, cv2.COLOR_RGB2BGR))

        # Update progress
        progress_bar.progress((i + 1) / frame_count)

    # Cleanup
    cap.release()
    out.release()
    os.remove("temp_video.mp4")

    # Display output
    st.write("Processing complete! Download the 3D anaglyph video below.")
    with open(output_path, "rb") as f:
        st.download_button("Download 3D Video", f, file_name="output_3d.mp4")
    os.remove(output_path)