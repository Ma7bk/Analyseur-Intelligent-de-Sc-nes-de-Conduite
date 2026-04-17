import os
import math
import json
import time
import requests
import streamlit as st
import numpy as np
import PIL.Image
from ultralytics import YOLO
from groq import Groq

# ==========================================
# MODULE A : MOTEUR DE DÉTECTION (YOLOv11s)
# ==========================================
class VisionSystem:
    def __init__(self, model_path='yolov11s.pt'):
        [span_5](start_span)"""Initialise YOLOv11s avec une résolution de 1280px[span_5](end_span)"""
        self.model = YOLO(model_path)
        [span_6](start_span)self.classes = ['car', 'truck', 'bus', 'pedestrian', 'rider', 'traffic sign', 'traffic light'] #[span_6](end_span)

    def process_image(self, image, conf=0.35, iou=0.45):
        [span_7](start_span)"""Exécute l'inférence avec NMS (Non-Maximum Suppression)[span_7](end_span)"""
        results = self.model.predict(
            source=image, 
            conf=conf, 
            iou=iou, 
            [span_8](start_span)imgsz=1280 # Optimisé pour les petits objets lointains[span_8](end_span)
        )
        return results[0]

# ==========================================
# MODULE B : AGENT INTELLIGENT (LLAMA 3.3)
# ==========================================
class HighwayAI:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        [span_9](start_span)self.focal_length = 1050 # Focale estimée pour capteur 720p[span_9](end_span)

    def tool_estimate_distance(self, obj_class, bbox_height):
        [span_10](start_span)"""Outil 04 : Triangulation géométrique[span_10](end_span)"""
        # Hauteurs réelles moyennes en mètres
        real_h = {"car": 1.5, "truck": 3.8, "bus": 3.5, "pedestrian": 1.7}
        h_m = real_h.get(obj_class, 1.6)
        
        # Formule : d = (H_réelle * Focale) / H_pixels
        distance = (h_m * self.focal_length) / bbox_height
        return round(distance, 2)

    def tool_safety_logic(self, speed, condition="sec"):
        [span_11](start_span)"""Outil 02 : Calcul selon Article R412-12[span_11](end_span)"""
        # [span_12](start_span)Coefficients de route[span_12](end_span)
        coeffs = {"sec": 1.0, "mouillé": 1.5, "brouillard": 2.0, "verglas": 3.0}
        c = coeffs.get(condition, 1.0)
        
        # Formule LaTeX pour ton rapport : d = (v / 3.6) * 2 * c
        d_conseillee = (speed / 3.6) * 2 * c
        return round(d_conseillee, 2)

    def generate_expert_report(self, detections, speed):
        [span_13](start_span)[span_14](start_span)"""Boucle ReAct : Analyse globale et décision de risque[span_13](end_span)[span_14](end_span)"""
        # Préparation du contexte pour LLAMA
        context = []
        for d in detections:
            dist = self.tool_estimate_distance(d['class'], d['h_px'])
            context.append(f"- {d['class']} détecté à {dist}m")

        system_prompt = """Tu es un expert en sécurité routière française. 
        Analyse les objets détectés, utilise les distances fournies et 
        détermine le niveau de risque : FAIBLE, MOYEN, ELEVE ou CRITIQUE. 
        [span_15](start_span)Réponds uniquement en format JSON strict."""[span_15](end_span)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Vitesse: {speed}km/h. Scène: {context}"}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

# ==========================================
# MODULE C : INTERFACE STREAMLIT (DASHBOARD)
# ==========================================
def main():
    st.set_page_config(page_title="IA Autoroute v5", layout="wide")
    st.sidebar.title("🛠 Configuration IA")
    
    # [span_16](start_span)[span_17](start_span)3 Onglets comme spécifié dans les slides[span_16](end_span)[span_17](end_span)
    tab1, tab2, tab3 = st.tabs(["Analyse Temps Réel", "Métriques Modèles", "Documentation"])

    with tab1:
        st.header("Analyseur de Scène")
        speed = st.sidebar.slider("Vitesse (km/h)", 50, 150, 120)
        uploaded = st.file_uploader("Image Dashcam...", type=['jpg', 'png'])

        if uploaded:
            # 1. Vision
            vision = VisionSystem()
            img = PIL.Image.open(uploaded)
            results = vision.process_image(img)
            
            # 2. IA
            ai = HighwayAI(api_key=st.secrets["GROQ_KEY"])
            detections = []
            for b in results.boxes:
                detections.append({
                    "class": vision.classes[int(b.cls)],
                    "h_px": b.xywh[0][3].item()
                })
            
            report = ai.generate_expert_report(detections, speed)
            
            # 3. Affichage
            c1, c2 = st.columns([2, 1])
            with c1:
                [span_18](start_span)st.image(results.plot(), caption="Détections YOLOv11s[span_18](end_span)")
            with c2:
                st.subheader("Rapport de Sécurité")
                st.json(report)
                risk = report.get("niveau_risque", "FAIBLE")
                st.progress(report.get("score_risque", 10) / 100)

    with tab2:
        st.header("Comparaison YOLOv8s vs YOLOv11s")
        # [span_19](start_span)Données issues de tes résultats finaux[span_19](end_span)
        st.table({
            "Critère": ["mAP@0.5", "Précision", "Époques", "imgsz"],
            "YOLOv8s": [0.515, 0.600, 78, 640],
            "YOLOv11s": [0.584, 0.671, 30, 1280]
        })
        [span_20](start_span)st.info("YOLOv11s montre une amélioration de +13.4% du mAP[span_20](end_span).")

if __name__ == "__main__":
    main()
