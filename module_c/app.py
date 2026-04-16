
import os
import sys
import json
import time
import tempfile
from datetime import datetime
from pathlib import Path
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import GROQ_API_KEY, OPENWEATHER_API_KEY, YOLO_MODEL_PATH
from module_b.agent_llm import HighwaySceneAgent, yolo_results_to_agent_format



st.set_page_config(
    page_title="Analyseur de Scènes Autoroutières",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.badge-FAIBLE   {background:#EAF3DE;color:#3B6D11;padding:5px 16px;border-radius:20px;font-weight:600;font-size:15px;display:inline-block}
.badge-MOYEN    {background:#FAEEDA;color:#854F0B;padding:5px 16px;border-radius:20px;font-weight:600;font-size:15px;display:inline-block}
.badge-ÉLEVÉ    {background:#FCEBEB;color:#A32D2D;padding:5px 16px;border-radius:20px;font-weight:600;font-size:15px;display:inline-block}
.badge-CRITIQUE {background:#A32D2D;color:white;  padding:5px 16px;border-radius:20px;font-weight:600;font-size:15px;display:inline-block}
</style>
""", unsafe_allow_html=True)


CLASS_COLORS = {
    "car":           (55, 138, 221),
    "truck":         (216, 90,  48),
    "bus":           (186, 117, 23),
    "pedestrian":    (29,  158, 117),
    "rider":         (83,  58,  183),
    "traffic sign":  (212, 83,  126),
    "traffic light": (226, 75,  74)
}

CLASS_NAMES_FR = {
    "car": "Voiture", "truck": "Camion", "bus": "Bus",
    "pedestrian": "Piéton", "rider": "Cycliste",
    "traffic sign": "Panneau", "traffic light": "Feu"
}

RISK_COLORS = {
    "FAIBLE":   "#639922",
    "MOYEN":    "#BA7517",
    "ÉLEVÉ":    "#E24B4A",
    "CRITIQUE": "#791F1F"
}



@st.cache_resource
def load_yolo_model(path: str):
    if not Path(path).exists():
        return None, f"Modèle non trouvé : {path}"
    try:
        from ultralytics import YOLO
        return YOLO(path), None
    except ImportError:
        return None, "ultralytics non installé. Lancer : pip install ultralytics"
    except Exception as e:
        return None, str(e)



def run_detection(model, image: Image.Image, conf: float, iou: float):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f.name)
        tmp_path = f.name

    t0 = time.time()
    results = model.predict(tmp_path, conf=conf, iou=iou, imgsz=1280, verbose=False)
    elapsed = round(time.time() - t0, 3)
    os.unlink(tmp_path)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    cls_map = {0:"car",1:"truck",2:"bus",3:"pedestrian",
               4:"rider",5:"traffic sign",6:"traffic light"}

    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf_val = float(box.conf[0])
        cls_name = cls_map.get(int(box.cls[0]), "?")
        color = CLASS_COLORS.get(cls_name, (136, 135, 128))
        # Boîte
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # Label
        label = f"{CLASS_NAMES_FR.get(cls_name, cls_name)} {conf_val:.0%}"
        lw = len(label) * 7 + 8
        draw.rectangle([x1, y1 - 22, x1 + lw, y1], fill=color)
        draw.text((x1 + 4, y1 - 18), label, fill=(255, 255, 255))

    detections = yolo_results_to_agent_format(results)
    detections["inference_time_s"] = elapsed
    return annotated, detections



def make_demo_report(detections: dict, speed_kmh: float) -> dict:
    objects = detections.get("objects", [])
    n = len(objects)
    trucks = sum(1 for o in objects if o["class"] == "truck")
    risk = ("CRITIQUE" if trucks > 2 else
            "ÉLEVÉ"    if trucks > 1 else
            "MOYEN"    if n > 3     else "FAIBLE")
    score = {"FAIBLE": 20, "MOYEN": 45, "ÉLEVÉ": 70, "CRITIQUE": 90}[risk]
    return {
        "niveau_risque": risk,
        "score_risque": score,
        "résumé_scène": (f"{n} objet(s) détecté(s) dont {trucks} poids lourd(s). "
                         f"Vitesse estimée : {speed_kmh} km/h."),
        "objets_détectés": {
            "véhicules": {
                "nombre": sum(1 for o in objects if o["class"] in ["car","truck","bus"]),
                "dont_poids_lourds": trucks,
                "proximité_immédiate": any(o["bbox"]["height"] > 0.2 for o in objects)
            },
            "piétons": sum(1 for o in objects if o["class"] == "pedestrian"),
            "panneaux": sum(1 for o in objects if o["class"] == "traffic sign"),
            "feux": sum(1 for o in objects if o["class"] == "traffic light")
        },
        "facteurs_risque": [
            f"{trucks} poids lourd(s) détecté(s)" if trucks else "Circulation normale",
            f"Vitesse élevée : {speed_kmh} km/h"
        ],
        "recommandations": [
            f"Maintenir une distance de {int(speed_kmh)} mètres",
            "Surveiller les angles morts des poids lourds" if trucks else "Rester vigilant"
        ],
        "vitesse_conseillée": f"{min(int(speed_kmh), 130)} km/h",
        "distance_sécurité_conseillée": f"{int(speed_kmh)} mètres",
        "actions_immédiates": (
            ["Réduire la vitesse immédiatement", "Augmenter la distance de sécurité"]
            if risk in ["ÉLEVÉ", "CRITIQUE"] else []
        ),
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "modèle_llm": "démonstration (YOLO non disponible)",
            "outils_appelés": 0,
            "vitesse_entrée_kmh": speed_kmh,
            "nb_objets_détectés": n
        }
    }



def risk_gauge(score: int, level: str) -> go.Figure:
    color = RISK_COLORS.get(level, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"<b>{level}</b>", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [
                {"range": [0,  25], "color": "#EAF3DE"},
                {"range": [25, 50], "color": "#FAEEDA"},
                {"range": [50, 75], "color": "#FCEBEB"},
                {"range": [75,100], "color": "#F7C1C1"}
            ]
        }
    ))
    fig.update_layout(
        height=210,
        margin=dict(t=50, b=5, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def detection_chart(detections: dict) -> go.Figure:
    counts = Counter(o["class"] for o in detections.get("objects", []))
    if not counts:
        return None
    labels = [CLASS_NAMES_FR.get(c, c) for c in counts]
    values = list(counts.values())
    colors = [f"rgb{CLASS_COLORS.get(c, (136,135,128))}" for c in counts]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=values, textposition="outside"
    ))
    fig.update_layout(
        title="Objets détectés",
        height=270,
        margin=dict(t=40, b=30, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Nombre"
    )
    return fig



def main():
    st.markdown("## 🛣️ Analyseur Intelligent de Scènes Autoroutières")
    st.markdown("*YOLOv8 + Agent LLM (Llama 3.3 via Groq) · Scénario : Autoroute*")
    st.divider()

    groq_ok  = bool(GROQ_API_KEY)
    meteo_ok = bool(OPENWEATHER_API_KEY)

    # ─────────────── SIDEBAR ───────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown(
            f"**Clé Groq (LLM)**  : {'✅ configurée' if groq_ok  else '❌ manquante'}  \n"
            f"**Clé Météo**        : {'✅ configurée' if meteo_ok else '⚠️ simulée'}"
        )
        if not groq_ok:
            st.error("Définir GROQ_API_KEY dans le terminal avant de lancer.")
        st.caption("Clés lues depuis les variables d'environnement — aucune saisie requise.")

        st.divider()
        st.markdown("### 🎚️ Détection YOLO")
        conf = st.slider(
            "Seuil de confiance", 0.10, 0.90, 0.35, 0.05,
            help="Un objet est retenu si le modèle est sûr à X% minimum"
        )
        iou = st.slider(
            "Seuil IoU (NMS)", 0.10, 0.90, 0.45, 0.05,
            help="Évite les détections en doublon sur le même objet"
        )

        st.divider()
        st.markdown("### 🚗 Contexte de conduite")
        speed = st.slider("Vitesse estimée (km/h)", 50, 150, 120, 10)

        use_gps = st.checkbox("Activer météo réelle (GPS)")
        location = None
        if use_gps:
            if not meteo_ok:
                st.warning("OPENWEATHER_API_KEY non configurée — météo simulée.")
            c1, c2 = st.columns(2)
            lat = c1.number_input("Latitude",  value=48.8566, format="%.4f")
            lon = c2.number_input("Longitude", value=2.3522,  format="%.4f")
            location = {"lat": lat, "lon": lon}

        st.divider()
        st.markdown("### ℹ️ Projet")
        st.markdown(
            "Module IA 2025–2026  \n"
            "Scénario : **Autoroute**  \n"
            "YOLOv8m + Llama 3.3 (Groq)"
        )

    tab1 = st.tabs(["📷 Analyse d'image"])[0]

    with tab1:
        col_up, col_demo = st.columns([3, 1])
        with col_up:
            uploaded = st.file_uploader(
                "Charger une image dashcam",
                type=["jpg", "jpeg", "png"],
                help="Photo prise par une caméra embarquée sur autoroute"
            )


        image = None
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

        if image is not None:
            st.image(image, use_container_width=True, caption="Image chargée")

            if st.button("🔍 Analyser la scène", type="primary", use_container_width=True):

                with st.spinner("⚙️ Détection YOLO en cours..."):
                    yolo_time = 0.0
                    if not Path(YOLO_MODEL_PATH).exists():
                        st.warning(
                            "Modèle YOLO non trouvé (`best_model_autoroute.pt`). "
                            "Le modèle sera disponible après l'entraînement sur Colab. "
                            "L'agent LLM s'exécute en mode démonstration."
                        )
                        annotated = image
                        detections = {
                            "objects": [], "image_width": image.width,
                            "image_height": image.height, "total_detections": 0,
                            "inference_time_s": 0.0
                        }
                    else:
                        model, err = load_yolo_model(YOLO_MODEL_PATH)
                        if err:
                            st.warning(f"Erreur YOLO : {err} — mode démonstration.")
                            annotated = image
                            detections = {
                                "objects": [], "image_width": image.width,
                                "image_height": image.height, "total_detections": 0,
                                "inference_time_s": 0.0
                            }
                        else:
                            annotated, detections = run_detection(model, image, conf, iou)
                            yolo_time = detections["inference_time_s"]
                            st.success(
                                f"✅ {detections['total_detections']} objet(s) détecté(s) "
                                f"en {yolo_time:.2f}s"
                            )

                with st.spinner("🤖 Agent LLM en cours d'analyse..."):
                    t0 = time.time()
                    if groq_ok:
                        try:
                            agent = HighwaySceneAgent()
                            report = agent.analyze_scene(
                                detections,
                                location=location,
                                speed_kmh=float(speed),
                                verbose=False
                            )
                        except Exception as e:
                            st.warning(f"Erreur agent LLM : {e} — mode démonstration.")
                            report = make_demo_report(detections, float(speed))
                    else:
                        st.warning("Clé Groq manquante — rapport de démonstration.")
                        report = make_demo_report(detections, float(speed))
                    llm_time = round(time.time() - t0, 2)

                st.divider()
                st.markdown("## 📋 Rapport d'analyse")

                col_img, col_info = st.columns([2, 1])

                with col_img:
                    st.image(annotated, use_container_width=True,
                             caption="Détections YOLO")

                with col_info:
                    level = report.get("niveau_risque", "MOYEN")
                    score = report.get("score_risque", 50)
                    st.markdown(
                        f'<div class="badge-{level}">⚠️ Risque : {level}</div>',
                        unsafe_allow_html=True
                    )
                    st.plotly_chart(
                        risk_gauge(score, level),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
                    st.metric("Vitesse conseillée",
                              report.get("vitesse_conseillée", "—"))
                    st.metric("Distance de sécurité",
                              report.get("distance_sécurité_conseillée", "—"))
                    st.metric("Temps total analyse",
                              f"{yolo_time + llm_time:.1f}s")

                st.markdown(f"**📝 Résumé :** {report.get('résumé_scène', '')}")

                chart = detection_chart(detections)
                if chart:
                    st.plotly_chart(chart, use_container_width=True,
                                    config={"displayModeBar": False})

                col_f, col_r = st.columns(2)
                with col_f:
                    st.markdown("#### ⚠️ Facteurs de risque")
                    for f in report.get("facteurs_risque", []):
                        st.markdown(f"- {f}")
                with col_r:
                    st.markdown("#### ✅ Recommandations")
                    for r in report.get("recommandations", []):
                        st.markdown(f"- {r}")

                # Actions immédiates
                actions = report.get("actions_immédiates", [])
                if actions and level in ["ÉLEVÉ", "CRITIQUE"]:
                    st.error("🚨 **ACTIONS IMMÉDIATES REQUISES**")
                    for a in actions:
                        st.markdown(f"🔴 {a}")

                with st.expander("🔍 Rapport JSON complet"):
                    st.json(report)
                with st.expander("📦 Détections YOLO brutes"):
                    st.json(detections)

                meta = report.get("metadata", {})
                st.caption(
                    f"Modèle LLM : {meta.get('modèle_llm', '—')} | "
                    f"Outils appelés : {meta.get('outils_appelés', 0)} | "
                    f"YOLO : {yolo_time:.2f}s | "
                    f"Agent LLM : {llm_time:.2f}s"
                )




if __name__ == "__main__":
    main()
