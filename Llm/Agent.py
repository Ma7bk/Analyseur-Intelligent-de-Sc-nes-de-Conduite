import math
import json
from groq import Groq # Client Groq v0.9.0

class HighwaySceneAgent:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192" # LLAMA 3.3 70B
        
    def get_weather_conditions(self, lat, lon):
        """Outil 01: Simule ou appelle OpenWeatherMap"""
        # Logique simplifiée pour la démo
        return {"condition": "pluie", "road_status": "mouillé", "coeff": 1.5}

    def calculate_safety_distance(self, speed, road_condition_coeff):
        """Outil 02: Article R412-12 du Code de la Route"""
        # Formule: (v/3.6) * 2 secondes * coefficient
        dist_sec = (speed / 3.6) * 2 * road_condition_coeff
        return round(dist_sec, 2)

    def get_highway_rules(self, situation):
        """Outil 03: Base de règles interne (Dict Python)"""
        rules = {
            "poids_lourds": "Article R413-2: Limite 110km/h, Angle mort 40m",
            "pluie": "Vitesse réduite à 110km/h sur autoroute",
            "urgence": "Arrêt sur bande d'arrêt d'urgence uniquement"
        }
        return rules.get(situation, "Règles générales du code de la route applicables")

    def estimate_object_distance(self, obj_class, bbox_h_px):
        """Outil 04: Triangulation géométrique"""
        # Constantes réelles
        real_heights = {"car": 1.5, "truck": 3.8, "bus": 3.5}
        focal_length = 1050 * (720 / 720) # Focal estimée pour 720p
        
        h_reelle = real_heights.get(obj_class, 1.6)
        distance = (h_reelle * focal_length) / bbox_h_px
        
        # Évaluation du risque de proximité
        risk = "NORMAL"
        if distance < 15: risk = "CRITIQUE"
        elif distance < 40: risk = "ELEVE"
        
        return {"distance_m": round(distance, 2), "risk_level": risk}

    def analyze_scene(self, detections, speed=120):
        """Boucle ReAct simplifiée générant le JSON final"""
        prompt = f"""
        En tant qu'expert sécurité routière, analyse ces détections YOLO: {detections}.
        Vitesse actuelle: {speed} km/h.
        Utilise tes outils pour évaluer le risque et donne une réponse en JSON STRICT.
        """
        # Ici l'agent appellerait les fonctions ci-dessus via l'API Groq (tool_use)
        # Pour le code, on simule la sortie finale structurée
        return {
            "niveau_risque": "ELEVE",
            "score_risque": 80,
            "recommandations": ["Réduire la vitesse", "Augmenter distance"],
            "vitesse_conseillee": 110
        }
