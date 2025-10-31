import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# ⚙️ CONFIGURACIÓN GENERAL
st.set_page_config(
    page_title="🔍 Detección de Objetos en Tiempo Real",
    page_icon="🤖",
    layout="wide"
)

# 🎨 ESTILOS PERSONALIZADOS (modo oscuro con neón)
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #e0e0e0;
}
.main {
    background-color: #0e1117;
    border-radius: 12px;
    padding: 1rem;
}
h1, h2, h3, h4 {
    color: #00ffc3;
    text-shadow: 0 0 15px #00ffc3;
}
.stButton>button {
    background-color: #00ffc3;
    color: #0e1117;
    border-radius: 8px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: scale(1.05);
    background-color: #00e6af;
}
.stSidebar {
    background-color: #111418;
}
.dataframe th {
    background-color: #00ffc3 !important;
    color: #0e1117 !important;
}
</style>
""", unsafe_allow_html=True)

# 🧠 FUNCIÓN PARA CARGAR MODELO YOLOv5
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.info("Prueba instalar YOLOv5 con: `pip install yolov5`")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
        except Exception as e2:
            st.error(f"No se pudo cargar desde Torch Hub: {e2}")
            return None

# 🧩 TÍTULO PRINCIPAL
st.title("🤖 Detección de Objetos con YOLOv5")
st.markdown("""
Bienvenido al laboratorio de visión computacional.  
Esta aplicación utiliza **YOLOv5** para detectar objetos en imágenes en tiempo real.  
Ajusta los parámetros y observa cómo el modelo identifica elementos visuales 🔍
""")

# ⚡ CARGA DEL MODELO
with st.spinner("🧠 Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# 🚀 INTERFAZ PRINCIPAL
if model:
    st.sidebar.title("🎚️ Parámetros de Detección")
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    
    with st.sidebar.expander("⚙️ Opciones avanzadas"):
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("Opciones avanzadas limitadas para este modelo")

    # 📸 CAPTURA DE IMAGEN
    st.markdown("### 📷 Captura de imagen o carga manual")
    col1, col2 = st.columns([2, 1])

    with col1:
        picture = st.camera_input("Toma una foto con tu cámara", key="camara_input")
    with col2:
        uploaded = st.file_uploader("O sube una imagen:", type=["jpg", "jpeg", "png"])

    if picture or uploaded:
        image_source = picture if picture else uploaded
        bytes_data = image_source.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # 🔍 DETECCIÓN
        with st.spinner("🛰️ Analizando imagen..."):
            try:
                results = model(cv2_img)
                results.render()
            except Exception as e:
                st.error(f"Error durante la detección: {e}")
                st.stop()

        # 📊 VISUALIZACIÓN DE RESULTADOS
        st.markdown("## 🧩 Resultados de Detección")

        col_img, col_data = st.columns(2)
        with col_img:
            st.image(cv2_img, channels="BGR", use_container_width=True, caption="🧠 Imagen procesada")
        
        with col_data:
            predictions = results.pred[0]
            if len(predictions) > 0:
                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                categories = predictions[:, 5]
                labels = model.names

                category_count = {}
                for c in categories:
                    idx = int(c.item())
                    category_count[idx] = category_count.get(idx, 0) + 1

                data = []
                for idx, count in category_count.items():
                    label = labels[idx]
                    conf = scores[categories == idx].mean().item()
                    data.append({
                        "Categoría": label,
                        "Cantidad": count,
                        "Confianza Promedio": f"{conf:.2f}"
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            else:
                st.info("No se detectaron objetos. Intenta reducir el umbral de confianza.")
else:
    st.error("🚨 No se pudo cargar el modelo YOLOv5. Verifica tu instalación.")
    st.stop()

# 💡 PIE DE PÁGINA
st.markdown("---")
st.markdown("""
### ⚙️ Sobre esta app
- Modelo: **YOLOv5 Small (pre-entrenado en COCO)**  
- Librerías: `torch`, `yolov5`, `streamlit`, `opencv`, `numpy`, `pandas`  
- Ejecuta detección de objetos en imágenes de cámara o carga manual  
""")
st.caption("🌌 Desarrollado con pasión por la visión artificial y el diseño elegante 🧠💫")
