import json
import requests
from typing import List, Dict, Any


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"  # change si besoin


class RiskAnalyzerLLM:
    def __init__(self, model_name: str = MODEL_NAME, api_url: str = OLLAMA_URL):
        self.model_name = model_name
        self.api_url = api_url

    def build_prompt(self, detections: List[Dict[str, Any]]) -> str:
        """
        Construit un prompt clair pour demander au LLM
        une analyse de risque à partir des objets détectés.
        """

        system_context = """
Tu es un assistant spécialisé en analyse de scènes routières.
Tu reçois une liste d'objets détectés dans une image de dashcam.

Chaque objet contient :
- class : type d'objet
- bbox : [x_center, y_center, width, height] normalisé entre 0 et 1
- confidence : score de confiance entre 0 et 1

Ta tâche :
1. Décrire brièvement la scène
2. Evaluer le niveau de risque : low, medium ou high
3. Donner la raison principale du risque
4. Retourner UNIQUEMENT un JSON valide au format :

{
  "scene_summary": "...",
  "risk_level": "low|medium|high",
  "main_reason": "...",
  "recommendation": "...",
  "critical_objects": [
    {
      "class": "...",
      "confidence": 0.0,
      "reason": "..."
    }
  ]
}
"""

        user_data = f"Détections :\n{json.dumps(detections, ensure_ascii=False, indent=2)}"

        return system_context.strip() + "\n\n" + user_data

    def call_llm(self, prompt: str) -> str:
        """
        Appelle le modèle via Ollama.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()
        return data.get("response", "").strip()

    def parse_response(self, raw_response: str) -> Dict[str, Any]:
        """
        Essaie de parser la réponse JSON du LLM.
        """
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "scene_summary": "Réponse non strictement JSON",
                "risk_level": "unknown",
                "main_reason": raw_response,
                "recommendation": "Vérifier le prompt ou renforcer la contrainte JSON.",
                "critical_objects": []
            }

    def analyze(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = self.build_prompt(detections)
        raw_response = self.call_llm(prompt)
        return self.parse_response(raw_response)


if __name__ == "__main__":
    # Exemple de sortie YOLO
    detections_example = [
        {
            "class": "car",
            "bbox": [0.52, 0.68, 0.22, 0.18],
            "confidence": 0.91
        },
        {
            "class": "pedestrian",
            "bbox": [0.48, 0.72, 0.08, 0.15],
            "confidence": 0.88
        },
        {
            "class": "traffic light",
            "bbox": [0.80, 0.20, 0.05, 0.10],
            "confidence": 0.79
        }
    ]

    analyzer = RiskAnalyzerLLM()
    result = analyzer.analyze(detections_example)

    print(json.dumps(result, ensure_ascii=False, indent=2))
