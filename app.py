import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# --- ตั้งค่าหน้าตาแอป ---
st.set_page_config(page_title="BacWise v3 - Bacterial Classifier", page_icon="🔬")

st.markdown("<h1 style='text-align: center;'>🔬 BacWise v3</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>ระบบจำแนกรูปร่างและสีแกรมแบคทีเรียอัตโนมัติ (โหลดโมเดลจากคลาวด์)</p>", unsafe_allow_html=True)
st.divider()

# --- ฟังก์ชันดาวน์โหลดไฟล์จาก Google Drive ---
@st.cache_resource
def download_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการดาวน์โหลด: {e}")
            return None
    return output_path

# --- ส่วนของการโหลดโมเดล ---
SHAPE_MODEL_ID = '18Mi_XotFtKUcPUzNCRWeq8CVdILJUpPa'
COLOR_MODEL_ID = '1ZCOyWd4Py43b2gUqPowgKGN5LIxNsGfQ'

with st.spinner('กำลังเชื่อมต่อกับ Google Drive เพื่อโหลดโมเดล...'):
    shape_path = download_model_from_drive(SHAPE_MODEL_ID, 'bacterial_shape_model_v3.h5')
    color_path = download_model_from_drive(COLOR_MODEL_ID, 'bacterial_color_model_v3.h5')
    
    if shape_path and color_path:
        model_shape = load_model(shape_path)
        model_color = load_model(color_path)
    else:
        st.error("❌ ไม่สามารถโหลดโมเดลได้ กรุณาตรวจสอบการตั้งค่าการแชร์ไฟล์ใน Google Drive ให้เป็น 'Anyone with the link'")
        st.stop()

# --- ฟังก์ชันพยากรณ์ ---
def predict_bacteria(img, model, labels):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    idx = np.argmax(preds)
    conf = preds[0][idx] * 100
    return labels[idx], conf

# --- ส่วนรับภาพจากผู้ใช้ ---
uploaded_file = st.file_uploader("เลือกรูปภาพแบคทีเรีย...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='รูปภาพที่อัปโหลด', use_container_width=True)
    
    with st.spinner('กำลังวิเคราะห์...'):
        shape_labels = ['Bacilli', 'Cocci', 'Spiral']
        color_labels = ['Gram Negative (Pink/Red)', 'Gram Positive (Purple/Blue)']
        
        res_shape, conf_shape = predict_bacteria(image, model_shape, shape_labels)
        res_color, conf_color = predict_bacteria(image, model_color, color_labels)
        
        st.success("วิเคราะห์เสร็จสิ้น!")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("รูปร่าง (Shape)")
            st.info(f"**{res_shape}**")
            st.write(f"ความมั่นใจ: {conf_shape:.2f}%")
        with col2:
            st.subheader("สีย้อม (Gram Stain)")
            text_color = "#FF69B4" if "Negative" in res_color else "#6A0DAD"
            st.markdown(f"<h3 style='color: {text_color};'>{res_color}</h3>", unsafe_allow_html=True)
            st.write(f"ความมั่นใจ: {conf_color:.2f}%")

st.divider()
st.caption("BacWise Project - พัฒนาโดยใช้ VGG16 Transfer Learning")