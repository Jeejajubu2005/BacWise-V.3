import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# --- 1. ตั้งค่าหัวข้อแอป (ใช้ดีไซน์ที่คุณทำใน Canva มาประกอบได้) ---
st.set_page_config(page_title="BacWise v.3", layout="centered")
st.title("🔬 BacWise: Bacterial Morphological Analysis")
st.write("ระบบวิเคราะห์รูปร่างและการย้อมแกรมแบคทีเรียอัตโนมัติ")

# --- 2. ฟังก์ชันดาวน์โหลดโมเดลจาก Google Drive (ใส่ File ID ของคุณ) ---
@st.cache_resource
def load_models():
    # แทนที่ 'YOUR_SHAPE_MODEL_ID' ด้วย ID จริงจาก Drive ของคุณ
    shape_id = 'YOUR_SHAPE_MODEL_ID' 
    color_id = 'YOUR_COLOR_MODEL_ID'
    
    shape_path = 'bacterial_shape_model_v3.h5'
    color_path = 'bacterial_color_model_v3.h5'

    if not os.path.exists(shape_path):
        gdown.download(f'https://drive.google.com/uc?id={shape_id}', shape_path, quiet=False)
    if not os.path.exists(color_path):
        gdown.download(f'https://drive.google.com/uc?id={color_id}', color_path, quiet=False)

    model_shape = tf.keras.models.load_model(shape_path)
    model_color = tf.keras.models.load_model(color_path)
    return model_shape, model_color

# โหลดโมเดล
try:
    model_shape, model_color = load_models()
    st.success("✅ โมเดลวิเคราะห์พร้อมใช้งาน!")
except Exception as e:
    st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")

# --- 3. ฟังก์ชันพยากรณ์ (จุดที่แก้ ValueError) ---
def predict_bacteria(image, model, labels, is_color=False):
    # 1. Resize ภาพให้เป็น 224x224 ตามที่ VGG16 ต้องการ
    img = image.resize((224, 224))
    # 2. แปลงเป็น Array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 3. Normalize (หาร 255.0)
    img_array = img_array / 255.0
    # 4. เพิ่มมิติ Batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. พยากรณ์
    preds = model.predict(img_array)
    
    # ดึงค่าที่แม่นยำที่สุด
    if is_color:
        # สำหรับโมเดลสี (Sigmoid - 1 output)
        confidence = preds[0][0]
        result_idx = 1 if confidence > 0.5 else 0
        final_conf = confidence if result_idx == 1 else 1 - confidence
    else:
        # สำหรับโมเดลรูปร่าง (Softmax - 3 outputs)
        result_idx = np.argmax(preds[0])
        final_conf = np.max(preds[0])
        
    return labels[result_idx], final_conf

# --- 4. ส่วนแสดงผลบนหน้าเว็บ ---
uploaded_file = st.file_uploader("อัปโหลดภาพถ่ายจากกล้องจุลทรรศน์...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="ภาพที่คุณอัปโหลด", use_container_width=True)
    
    if st.button("เริ่มการวิเคราะห์"):
        with st.spinner('AI กำลังวิเคราะห์...'):
            # กำหนด Label ให้ตรงกับโฟลเดอร์ที่คุณเทรน
            shape_labels = ['Bacilli', 'Cocci', 'Spiral']
            color_labels = ['Gram Negative', 'Gram Positive'] # เช็คลำดับให้ตรงกับ class_indices ใน Colab

            # พยากรณ์รูปร่าง
            res_shape, conf_shape = predict_bacteria(image, model_shape, shape_labels)
            # พยากรณ์สี
            res_color, conf_color = predict_bacteria(image, model_color, color_labels, is_color=True)

            # แสดงผลลัพธ์
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("รูปร่าง (Shape)", res_shape)
                st.write(f"ความมั่นใจ: {conf_shape*100:.2f}%")
            
            with col2:
                # เปลี่ยนสีตัวอักษรตามผล Gram
                color_hex = "#FF69B4" if "Negative" in res_color else "#8A2BE2"
                st.subheader("การย้อมสี (Gram)")
                st.markdown(f"<h2 style='color: {color_hex};'>{res_color}</h2>", unsafe_allow_html=True)
                st.write(f"ความมั่นใจ: {conf_color*100:.2f}%")

            # คำแนะนำเพิ่มเติม
            st.info("💡 คำแนะนำ: หากภาพมีเชื้อเกาะกลุ่มกันแน่น ผลการวิเคราะห์รูปร่างอาจคลาดเคลื่อนได้")
