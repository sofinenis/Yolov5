import cv2
import streamlit as st
import numpy as np
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# 🌻 Configuración de la página
st.set_page_config(
    page_title="🌻 Detección de Objetos entre Girasoles",
    page_icon="🌻",
    layout="wide"
)

# 🌻 Estilos personalizados
st.markdown("""
<style>
body {
    background-color: #fff8dc;
    color: #4a3000;
}
.main {
    background-color: #fffbea;
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0px 0px 25px #f1c40f50;
}
h1, h2, h3 {
    color: #d4a017;
    text-align: center;
    font-family: 'Comic Sans MS', cursive;
}
.stButton>button {
    background-color: #f4d03f;
    color: #4a3000;
    border-radius: 10px;
    border: 2px solid #d4a017;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #f1c40f;
    transform: scale(1.05);
    border-color: #b8860b;
}
.stSidebar {
    background-color: #fff8dc !important;
}
.dataframe th {
    background-color: #f9e79f !important;
    color: #4a3000 !important;
}
</style>
""", unsafe_allow_html=True)

# 🌻 Función para cargar el modelo YOLOv5
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
        st.error(f"🌻 No se pudo cargar el modelo: {e}")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
        except Exception as e2:
            st.error(f"Error alternativo: {e2}")
            return None

# 🌻 Título principal
st.title("🌻 Detección de Objetos entre Girasoles 🌞")
st.markdown("""
Imagina que estás en un campo de girasoles mientras una inteligencia artificial 🌼  
observa cada detalle y detecta los objetos que te rodean 🌻✨  
Utiliza **YOLOv5** para reconocer elementos en tus imágenes.
""")

# 🌻 Carga del modelo
with st.spinner("🌻 Cargando el modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    st.sidebar.title("🌼 Parámetros de Detección")
    model.conf = st.sidebar.slider('Nivel de confianza 🌞', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU 🌻', 0.0, 1.0, 0.45, 0.01)

    with st.sidebar.expander("🌸 Opciones avanzadas"):
        try:
            model.agnostic = st.checkbox('NMS class-agnostic', False)
            model.multi_label = st.checkbox('Múltiples etiquetas por caja', False)
            model.max_det = st.number_input('Detecciones máximas', 10, 2000, 1000, 10)
        except:
            st.warning("🌻 Algunas opciones no están disponibles.")

    # 🌻 Captura de imagen o carga manual
    st.markdown("### 📸 Toma una foto o sube una imagen 🌼")
    col1, col2 = st.columns([2, 1])

    with col1:
        picture = st.camera_input("Captura con tu cámara 🌻", key="camara_input")

    with col2:
        uploaded = st.file_uploader("O sube una imagen desde tu campo de girasoles 🌞", type=["jpg", "jpeg", "png"])

    if picture or uploaded:
        image_source = picture if picture else uploaded
        bytes_data = image_source.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # 🌻 Detección
        with st.spinner("🌻 Analizando imagen... floreciendo resultados 🌼"):
            try:
                results = model(cv2_img)
                results.render()
            except Exception as e:
                st.error(f"Error durante la detección: {e}")
                st.stop()

        # 🌼 Visualización
        st.markdown("## 🌞 Resultados del campo de visión 🌻")

        col_img, col_data = st.columns(2)
        with col_img:
            st.image(cv2_img, channels="BGR", use_container_width=True, caption="🌻 Imagen procesada con YOLOv5 🌼")

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
                        "Categoría 🌻": label,
                        "Cantidad 🌼": count,
                        "Confianza Promedio 🌞": f"{conf:.2f}"
                    })

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría 🌻")["Cantidad 🌼"])
            else:
                st.info("🌻 No se detectaron objetos. ¡Prueba con otra flor! 🌼")

else:
    st.error("🌻 No se pudo cargar el modelo YOLOv5. Verifica tu instalación e inténtalo de nuevo.")
    st.stop()

# 🌻 Pie de página
st.markdown("---")
st.markdown("""
### 🌼 Acerca de esta aplicación
Esta versión floreciente usa **YOLOv5** para detectar objetos en imágenes,
con un toque de **alegría y girasoles** 🌻.  
Desarrollado con cariño, sol y código por ti 🌞.
""")
st.caption("🌻 La tecnología también puede florecer 🌼")
