#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
import streamlit as st
from ultralytics import YOLO

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="Yoga Pose Estimation using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# main page heading
st.title("Yoga Pose Estimation using YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Pose Estimation"]
)

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model = YOLO(r'best.pt')

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

with st.container(border=True):
    source_img = None
    if source_selectbox == config.SOURCES_LIST[0]:  # Image
        infer_uploaded_image(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]:  # Video
        infer_uploaded_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[2]:  # Webcam
        infer_uploaded_webcam(confidence, model)
    else:
        st.error("Currently only 'Image' and 'Video' source are implemented")
