"""
PROJET : Analyseur Intelligent de Scènes Autoroutières
MODULE : Système Expert Intégré (Module B & C)
AUTEUR : Zakaria (zakariabel1)
VERSION : 2.0 - Mise à jour majeure de la logique de risque
"""

import math
import json
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SafetyExpertZakaria:
    """
    Système expert pour l'évaluation multicritère du risque routier.
    Implémente les calculs de distance géométrique et les règles du Code de la Route.
    """
    
    def __init__(self, speed_kmh=120, road_condition="sec"):
        self.speed = speed_kmh
        self.condition = road_condition
        # [span_0](start_span)Focal estimée pour images 1280x720[span_0](end_span)
        self.focal_length = 1050 
        # [span_1](start_span)Hauteurs réelles standards (en mètres)[span_1](end_span)
        self.real_heights = {
            "car": 1.5,
            "truck": 3.8,
            "bus": 3.5,
            "pedestrian": 1.7,
            "rider": 1.6,
            "traffic sign": 0.8,
            "traffic light": 2.5
        }

    def get_safety_coeff(self):
        [span_2](start_span)[span_3](start_span)"""Retourne le coefficient multiplicateur selon l'état de la route[span_2](end_span)[span_3](end_span)"""
        coeffs = {
            "sec": 1.0,
            "mouillé": 1.5,
            "brouillard": 2.0,
            "verglas": 3.0
        }
        return coeffs.get(self.condition, 1.0)

    def calculate_legal_distance(self):
        """
        [span_4](start_span)Calcule la distance de sécurité légale selon l'article R412-12[span_4](end_span).
        Formule : d = (v / 3.6) * 2 * coeff
        """
        v_ms = self.speed / 3.6
        return round(v_ms * 2 * self.get_safety_coeff(), 2)

    def triangulate_distance(self, obj_class, bbox_height_px):
        """
        [span_5](start_span)Estime la distance réelle d'un objet via triangulation[span_5](end_span).
        $$d = \frac{H_{reelle} \times f}{H_{pixels}}$$
        """
        h_real = self.real_heights.get(obj_class, 1.6)
        if bbox_height_px <= 0: return 999
        
        distance = (h_real * self.focal_length) / bbox_height_px
        return round(distance, 2)

    def evaluate_global_risk(self, detections):
        """
        Analyse une liste de détections YOLO et génère un score de risque sur 100.
        """
        risk_score = 0
        factors = []
        
        legal_dist = self.calculate_legal_distance()
        
        for det in detections:
            obj_class = det['class']
            h_px = det['h_px']
            dist = self.triangulate_distance(obj_class, h_px)
            
            # [span_6](start_span)Alerte si un camion est trop proche[span_6](end_span)
            if obj_class == "truck" and dist < 20:
                risk_score += 40
                factors.append(f"CRITIQUE : Camion à {dist}m (Angle mort dangereux)")
            
            # Alerte si distance inférieure à la distance légale
            if dist < legal_dist:
                risk_score += 20
                factors.append(f"ALERTE : {obj_class} à {dist}m (Légal min: {legal_dist}m)")
                
        # Plafonnement
        risk_score = min(risk_score, 100)
        
        return {
            "score": risk_score,
            "level": "CRITIQUE" if risk_score > 75 else "ELEVE" if risk_score > 50 else "MOYEN" if risk_score > 25 else "FAIBLE",
            "factors": factors,
            "legal_ref": "Article R412-12 & R413-2"
        }

# --- PARTIE VISUALISATION (Pour le Module C) ---
def create_risk_gauge(score):
    """Génère une jauge de risque interactive avec Plotly"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Niveau de Risque Global", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 25], 'color': "green"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score}}))
    return fig

def create_detection_chart(detections):
    """Génère un graphique à barres des objets détectés"""
    if not detections: return None
    df = pd.DataFrame(detections)
    counts = df['class'].value_counts().reset_index()
    counts.columns = ['Classe', 'Nombre']
    fig = px.bar(counts, x='Classe', y='Nombre', title="Répartition des objets sur la scène", color='Classe')
    return fig
