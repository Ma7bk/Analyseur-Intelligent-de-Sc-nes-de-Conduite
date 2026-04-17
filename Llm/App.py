import streamlit as st
[span_17](start_span)from ultralytics import YOLO # Framework YOLO[span_17](end_span)
from agent_llm import HighwaySceneAgent
import PIL.Image

# [span_18](start_span)Configuration de la page[span_18](end_span)
st.set_page_config(page_title="Analyseur IA Autoroute", layout="wide")

st.title("🚗 Analyseur Intelligent de Scènes Autoroutières")

# [span_19](start_span)Sidebar pour les réglages[span_19](end_span)
with st.sidebar:
    st.header("Configuration")
    [span_20](start_span)conf_threshold = st.slider("Seuil de confiance YOLO", 0.0, 1.0, 0.35)[span_20](end_span)
    [span_21](start_span)speed_input = st.slider("Vitesse estimée (km/h)", 0, 150, 120)[span_21](end_span)
    use_weather = st.checkbox("Activer météo réelle (GPS)")

# [span_22](start_span)Chargement du modèle[span_22](end_span)
@st.cache_resource
def load_model():
    [span_23](start_span)return YOLO("yolov11s.pt") # Version supérieure v11s[span_23](end_span)

model = load_model()
agent = HighwaySceneAgent(api_key="GROQ_API_KEY")

# [span_24](start_span)Upload d'image[span_24](end_span)
uploaded_file = st.file_uploader("Upload une image de Dashcam", type=['jpg', 'png'])

if uploaded_file:
    img = PIL.Image.open(uploaded_file)
    
    # [span_25](start_span)Étape 01: Détection YOLO[span_25](end_span)
    [span_26](start_span)results = model.predict(source=img, conf=conf_threshold, imgsz=1280)[span_26](end_span)
    
    # [span_27](start_span)Préparation des données pour l'agent[span_27](end_span)
    detections_for_llm = []
    for box in results[0].boxes:
        cls = model.names[int(box.cls)]
        h_px = box.xywh[0][3].item() # Hauteur de la Bbox
        detections_for_llm.append({"class": cls, "h_px": h_px})

    # [span_28](start_span)Étape 02: Analyse Agent LLM[span_28](end_span)
    with st.spinner("L'IA analyse la situation..."):
        analysis = agent.analyze_scene(detections_for_llm, speed=speed_input)
    
    # [span_29](start_span)Étape 03: Affichage Dashboard[span_29](end_span)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Détection Visuelle")
        [span_30](start_span)st.image(results[0].plot()) # Image annotée par YOLO[span_30](end_span)
        
    with col2:
        st.subheader("Rapport d'Analyse IA")
        st.metric("Niveau de Risque", analysis["niveau_risque"], delta=analysis["score_risque"])
        st.write(f"**Vitesse conseillée :** {analysis['vitesse_conseillee']} km/h")
        st.write("**Actions immédiates :**")
        for rec in analysis["recommandations"]:
            st.error(rec) if analysis["score_risque"] > 70 else st.warning(rec)
