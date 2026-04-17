import os
import sys
import json
import requests
from datetime import datetime
from typing import Optional
from groq import Groq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GROQ_API_KEY, OPENWEATHER_API_KEY, MODEL_NAME

SYSTEM_PROMPT = """Tu es un système expert en analyse de sécurité routière pour la conduite autoroutière.
Tu reçois les résultats d'un modèle de détection d'objets (YOLOv8) appliqué à des images de dashcam.

## Contexte : Autoroute
- Vitesses élevées (90-130 km/h) — distances de sécurité critiques
- Peu de piétons, dominance de véhicules légers et poids lourds
- Risques principaux : changement de voie, dépassement, véhicule lent
- Code de la route français applicable

## Niveaux de risque
- FAIBLE   : circulation fluide, bonne visibilité
- MOYEN    : densité modérée, points d'attention
- ÉLEVÉ    : situation dangereuse, action recommandée
- CRITIQUE : danger imminent

## Format de réponse — JSON UNIQUEMENT, sans texte autour
{
  "niveau_risque": "FAIBLE|MOYEN|ÉLEVÉ|CRITIQUE",
  "score_risque": 0-100,
  "résumé_scène": "1-2 phrases",
  "objets_détectés": {
    "véhicules": {"nombre": N, "dont_poids_lourds": N, "proximité_immédiate": true/false},
    "piétons": N,
    "panneaux": N,
    "feux": N
  },
  "facteurs_risque": ["facteur1", "facteur2"],
  "recommandations": ["reco1", "reco2"],
  "vitesse_conseillée": "X km/h",
  "distance_sécurité_conseillée": "X mètres",
  "actions_immédiates": ["action1"]
}"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_conditions",
            "description": "Récupère les conditions météo actuelles via OpenWeatherMap. Adapte les recommandations selon pluie/brouillard/verglas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude":  {"type": "number", "description": "Latitude GPS"},
                    "longitude": {"type": "number", "description": "Longitude GPS"}
                },
                "required": ["latitude", "longitude"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_safety_distance",
            "description": "Calcule la distance de sécurité selon vitesse et conditions. Code de la route français R412-12.",
            "parameters": {
                "type": "object",
                "properties": {
                    "speed_kmh":      {"type": "number", "description": "Vitesse en km/h"},
                    "road_condition": {"type": "string", "enum": ["sec", "mouillé", "verglas", "brouillard"]},
                    "vehicle_type":   {"type": "string", "enum": ["voiture", "poids_lourd"]}
                },
                "required": ["speed_kmh", "road_condition"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_highway_rules",
            "description": "Consulte la base de règles du code de la route français pour autoroutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "situation": {
                        "type": "string",
                        "enum": ["dépassement", "insertion", "poids_lourds",
                                 "vitesse_limite", "distance_sécurité",
                                 "conditions_météo", "urgence"]
                    }
                },
                "required": ["situation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_object_distance",
            "description": "Estime la distance d'un objet détecté par triangulation géométrique à partir de sa bounding box.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_height_px":  {"type": "number", "description": "Hauteur bounding box en pixels"},
                    "image_height_px": {"type": "number", "description": "Hauteur image en pixels"},
                    "object_class":    {"type": "string",
                                       "enum": ["car", "truck", "bus", "pedestrian", "traffic sign"]}
                },
                "required": ["bbox_height_px", "image_height_px", "object_class"]
            }
        }
    }
]



def get_weather_conditions(latitude: float, longitude: float) -> dict:
    if not OPENWEATHER_API_KEY:
        return {
            "condition": "dégagé",
            "temperature": 18,
            "humidity": 45,
            "wind_speed_kmh": 12,
            "visibility_km": 10,
            "road_condition_estimate": "sec",
            "source": "simulé — définir OPENWEATHER_API_KEY pour données réelles"
        }
    try:
        url = (f"https://api.openweathermap.org/data/2.5/weather"
               f"?lat={latitude}&lon={longitude}"
               f"&appid={OPENWEATHER_API_KEY}&units=metric&lang=fr")
        data = requests.get(url, timeout=5).json()
        wid = data["weather"][0]["id"]
        road = ("mouillé"   if 500 <= wid < 532 else
                "verglas"   if 600 <= wid < 623 else
                "brouillard" if 700 <= wid < 772 else "sec")
        return {
            "condition": data["weather"][0]["description"],
            "temperature": round(data["main"]["temp"], 1),
            "humidity": data["main"]["humidity"],
            "wind_speed_kmh": round(data["wind"]["speed"] * 3.6, 1),
            "visibility_km": data.get("visibility", 10000) / 1000,
            "road_condition_estimate": road,
            "source": "OpenWeatherMap"
        }
    except Exception as e:
        return {"error": str(e), "road_condition_estimate": "sec", "source": "erreur API"}


def calculate_safety_distance(speed_kmh: float, road_condition: str,
                               vehicle_type: str = "voiture") -> dict:
    base = (speed_kmh / 3.6) * 2
    mult = {"sec": 1.0, "mouillé": 1.5, "brouillard": 2.0, "verglas": 3.0}.get(road_condition, 1.0)
    if vehicle_type == "poids_lourd":
        mult *= 1.3
    recommended = max(base * mult, speed_kmh)
    return {
        "distance_recommandée_m": round(recommended),
        "distance_légale_minimum_m": round(speed_kmh),
        "condition": road_condition,
        "multiplicateur": mult,
        "article": "R412-12",
        "calcul": f"{speed_kmh} km/h × {mult} = {round(recommended)}m"
    }


def get_highway_rules(situation: str) -> dict:
    rules = {
        "dépassement": {
            "règles": [
                "Dépasser uniquement par la gauche",
                "Interdiction de dépasser par la droite sauf embouteillage",
                "Reprendre la voie de droite après dépassement",
                "Signaler avec le clignotant"
            ], "article": "R414-4"
        },
        "poids_lourds": {
            "règles": [
                "PL > 3.5t limités à 110 km/h",
                "Distance de sécurité doublée derrière un PL",
                "Angle mort de 40m à l'arrière",
                "Ne pas rester dans l'angle mort"
            ], "article": "R413-2"
        },
        "vitesse_limite": {
            "règles": [
                "130 km/h par temps sec",
                "110 km/h par temps de pluie",
                "80 km/h minimum voie gauche"
            ], "article": "R413-2"
        },
        "distance_sécurité": {
            "règles": [
                "Distance = vitesse exprimée en mètres (130 km/h → 130m)",
                "Doubler sur sol mouillé",
                "Tripler sur verglas"
            ], "article": "R412-12"
        },
        "conditions_météo": {
            "règles": [
                "Pluie → 110 km/h maximum",
                "Brouillard < 50m → 50 km/h maximum",
                "Allumer feux brouillard arrière si visibilité < 50m"
            ], "article": "R416-5"
        },
        "insertion": {
            "règles": [
                "Céder le passage aux véhicules en circulation",
                "Accélérer sur la voie d'accélération",
                "Signaler avec le clignotant"
            ], "article": "R415-7"
        },
        "urgence": {
            "règles": [
                "S'arrêter uniquement sur la BAU",
                "Triangle de signalisation à 100m minimum",
                "Gilet jaune obligatoire avant de sortir"
            ], "article": "R416-2"
        }
    }
    return rules.get(situation, {"règles": ["Situation non répertoriée"], "article": "N/A"})


def estimate_object_distance(bbox_height_px: float, image_height_px: float,
                              object_class: str) -> dict:
    real_heights = {
        "car": 1.5, "truck": 3.8, "bus": 3.5,
        "pedestrian": 1.75, "traffic sign": 0.9
    }
    focal = 1050 * (image_height_px / 720)
    if bbox_height_px <= 0:
        return {"error": "bbox_height_px doit être > 0"}
    dist = (real_heights.get(object_class, 1.5) * focal) / bbox_height_px
    proximity = ("très proche — danger immédiat" if dist < 15 else
                 "proche — attention requise"    if dist < 40 else
                 "distance modérée"              if dist < 80 else
                 "loin — surveillance")
    return {
        "distance_estimée_m": round(dist, 1),
        "proximité": proximity,
        "objet": object_class,
        "méthode": "triangulation géométrique",
        "précision": "±20%"
    }


TOOL_FUNCTIONS = {
    "get_weather_conditions":    get_weather_conditions,
    "calculate_safety_distance": calculate_safety_distance,
    "get_highway_rules":         get_highway_rules,
    "estimate_object_distance":  estimate_object_distance,
}


def execute_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return json.dumps({"error": f"Outil inconnu : {name}"})
    try:
        return json.dumps(func(**args), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})




class HighwaySceneAgent:


    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY manquante.\n"
                "Définir dans le terminal :\n"
                "  Windows : set GROQ_API_KEY=gsk_...\n"
                "  Linux   : export GROQ_API_KEY=gsk_..."
            )
        self.client = Groq(api_key=GROQ_API_KEY)

    def analyze_scene(self, detections: dict, location: Optional[dict] = None,
                      speed_kmh: float = 120.0, verbose: bool = True) -> dict:

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": self._build_prompt(detections, location, speed_kmh)}
        ]

        if verbose:
            print(f"\n{'='*55}")
            print("AGENT — Analyse de scène autoroutière")
            print(f"{'='*55}")
            print(f"Objets : {len(detections.get('objects', []))} | Vitesse : {speed_kmh} km/h")

        tool_count = 0

        while tool_count < 8:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=2000,
                temperature=0.1
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    tool_count += 1
                    result = execute_tool(name, args)
                    if verbose:
                        print(f"\n  Outil    : {name}")
                        print(f"  Args     : {json.dumps(args, ensure_ascii=False)}")
                        print(f"  Résultat : {result[:120]}...")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    })

            else:
                if verbose:
                    print(f"\nOutils appelés : {tool_count} | Rapport généré.")
                raw = (msg.content or "{}").strip()
                if raw.startswith("```"):
                    parts = raw.split("```")
                    raw = parts[1] if len(parts) > 1 else raw
                    if raw.startswith("json"):
                        raw = raw[4:]
                try:
                    report = json.loads(raw.strip())
                except json.JSONDecodeError:
                    report = {
                        "niveau_risque": "MOYEN",
                        "score_risque": 50,
                        "résumé_scène": raw[:300],
                        "erreur": "Réponse LLM non parseable en JSON"
                    }
                report["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "modèle_llm": MODEL_NAME,
                    "outils_appelés": tool_count,
                    "vitesse_entrée_kmh": speed_kmh,
                    "nb_objets_détectés": len(detections.get("objects", []))
                }
                return report

        return {
            "niveau_risque": "MOYEN", "score_risque": 50,
            "résumé_scène": "Analyse incomplète.",
            "erreur": "Limite d'appels outils atteinte"
        }

    def _build_prompt(self, detections: dict, location: Optional[dict],
                      speed_kmh: float) -> str:
        objects = detections.get("objects", [])
        counts = {}
        for o in objects:
            cls = o.get("class", "?")
            counts[cls] = counts.get(cls, 0) + 1
        close = [o for o in objects
                 if o.get("confidence", 0) > 0.7
                 and o.get("bbox", {}).get("height", 0) > 0.2]

        return f"""## Scène autoroutière à analyser

Vitesse estimée : {speed_kmh} km/h
Heure           : {datetime.now().strftime('%H:%M')}
Position GPS    : {f"lat={location['lat']}, lon={location['lon']}" if location else "non fournie"}

Objets détectés ({len(objects)}) :
{json.dumps(counts, ensure_ascii=False)}

Détails (confiance > 0.5) :
{json.dumps([o for o in objects if o.get('confidence', 0) > 0.5], ensure_ascii=False, indent=2)}

Objets proches (bbox > 20% image) : {len(close)}
Résolution image : {detections.get('image_width', 1280)}x{detections.get('image_height', 720)} px

Instructions pour l'analyse :
1. Si position GPS fournie → appeler get_weather_conditions
2. Pour chaque objet proche → appeler estimate_object_distance
3. Appeler calculate_safety_distance avec la condition route
4. Appeler get_highway_rules si situation particulière détectée
5. Retourner UNIQUEMENT le JSON final"""




def yolo_results_to_agent_format(yolo_results) -> dict:
    """Convertit model.predict() vers le format attendu par HighwaySceneAgent."""
    CLASS_NAMES = {
        0: "car", 1: "truck", 2: "bus", 3: "pedestrian",
        4: "rider", 5: "traffic sign", 6: "traffic light"
    }
    result = yolo_results[0]
    img_h, img_w = result.orig_shape[:2]
    objects = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        objects.append({
            "class": CLASS_NAMES.get(cls_id, "unknown"),
            "class_id": cls_id,
            "confidence": round(conf, 3),
            "bbox": {
                "x1": round(x1), "y1": round(y1),
                "x2": round(x2), "y2": round(y2),
                "width":    round((x2 - x1) / img_w, 4),
                "height":   round((y2 - y1) / img_h, 4),
                "x_center": round((x1 + x2) / (2 * img_w), 4),
                "y_center": round((y1 + y2) / (2 * img_h), 4)
            }
        })
    objects.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "objects": objects,
        "image_width": img_w,
        "image_height": img_h,
        "total_detections": len(objects)
    }




if __name__ == "__main__":
    test_detections = {
        "objects": [
            {
                "class": "car", "class_id": 0, "confidence": 0.94,
                "bbox": {"x1": 450, "y1": 280, "x2": 680, "y2": 420,
                         "width": 0.18, "height": 0.19, "x_center": 0.44, "y_center": 0.49}
            },
            {
                "class": "truck", "class_id": 1, "confidence": 0.89,
                "bbox": {"x1": 200, "y1": 200, "x2": 550, "y2": 460,
                         "width": 0.27, "height": 0.36, "x_center": 0.30, "y_center": 0.46}
            },
            {
                "class": "car", "class_id": 0, "confidence": 0.76,
                "bbox": {"x1": 750, "y1": 310, "x2": 900, "y2": 400,
                         "width": 0.12, "height": 0.13, "x_center": 0.65, "y_center": 0.49}
            },
            {
                "class": "traffic sign", "class_id": 5, "confidence": 0.82,
                "bbox": {"x1": 50, "y1": 150, "x2": 130, "y2": 250,
                         "width": 0.06, "height": 0.14, "x_center": 0.07, "y_center": 0.28}
            }
        ],
        "image_width": 1280, "image_height": 720, "total_detections": 4
    }

    agent = HighwaySceneAgent()
    report = agent.analyze_scene(
        test_detections,
        location={"lat": 48.8566, "lon": 2.3522},
        speed_kmh=120,
        verbose=True
    )
    print("\n" + "="*55)
    print("RAPPORT FINAL")
    print("="*55)
    print(json.dumps(report, ensure_ascii=False, indent=2))
