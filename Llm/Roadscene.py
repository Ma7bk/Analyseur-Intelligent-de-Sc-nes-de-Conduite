[
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import requests


class RoadSceneLLMAnalyzer:
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        api_url: str = "http://localhost:11434/api/generate",
        output_dir: str = "outputs",
    ) -> None:
        self.model_name = model_name
        self.api_url = api_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_priority_score(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Heuristique simple avant appel LLM.
        Plus un objet est bas/central dans l'image, plus il est potentiellement proche.
        """
        vulnerable_classes = {"pedestrian", "rider"}
        vehicle_classes = {"car", "truck", "bus"}

        score = 0
        critical_objects = []

        for obj in detections:
            cls = obj.get("class", "").lower()
            conf = float(obj.get("confidence", 0.0))
            bbox = obj.get("bbox", [0, 0, 0, 0])

            if len(bbox) != 4:
                continue

            x_center, y_center, width, height = bbox

            proximity_score = 0
            if y_center > 0.6:
                proximity_score += 2
            if 0.35 <= x_center <= 0.65:
                proximity_score += 2
            if width * height > 0.03:
                proximity_score += 2

            class_score = 0
            if cls in vulnerable_classes:
                class_score += 4
            elif cls in vehicle_classes:
                class_score += 2
            elif cls in {"light", "traffic sign"}:
                class_score += 1

            object_score = (proximity_score + class_score) * conf
            score += object_score

            if object_score >= 3:
                critical_objects.append(
                    {
                        "class": cls,
                        "confidence": round(conf, 3),
                        "bbox": bbox,
                        "object_score": round(object_score, 3),
                    }
                )

        if score >= 10:
            risk_hint = "high"
        elif score >= 5:
            risk_hint = "medium"
        else:
            risk_hint = "low"

        return {
            "heuristic_score": round(score, 3),
            "heuristic_risk_hint": risk_hint,
            "critical_objects": critical_objects,
        }

    def build_prompt(self, detections: List[Dict[str, Any]], heuristic: Dict[str, Any]) -> str:
        return f"""
Tu es un assistant expert en analyse de scènes routières pour dashcam.

Tu reçois :
1. une liste de détections d'objets issues de YOLO
2. un score heuristique calculé automatiquement

Ton rôle :
- analyser la scène,
- estimer le niveau de risque global,
- expliquer pourquoi,
- proposer une action recommandée.

Contraintes strictes :
- Réponds UNIQUEMENT avec un JSON valide.
- N'ajoute aucun texte avant ou après le JSON.
- Le champ risk_level doit être exactement : "low", "medium" ou "high".

Format JSON attendu :
{{
  "scene_summary": "string",
  "risk_level": "low|medium|high",
  "risk_score": 0,
  "main_reason": "string",
  "recommended_action": "string",
  "objects_of_interest": [
    {{
      "class": "string",
      "confidence": 0.0,
      "reason": "string"
    }}
  ]
}}

Détections YOLO :
{json.dumps(detections, ensure_ascii=False, indent=2)}

Pré-analyse heuristique :
{json.dumps(heuristic, ensure_ascii=False, indent=2)}
""".strip()

    def call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def extract_json(self, raw_text: str) -> Dict[str, Any]:
        """
        Tente d'extraire un JSON même si le modèle ajoute du texte autour.
        """
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        return {
            "scene_summary": "Impossible de parser la réponse du LLM.",
            "risk_level": "medium",
            "risk_score": 50,
            "main_reason": raw_text,
            "recommended_action": "Vérification manuelle requise.",
            "objects_of_interest": [],
        }

    def save_result(self, result: Dict[str, Any], filename: str = "llm_analysis.json") -> Path:
        file_path = self.output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return file_path

    def analyze(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        heuristic = self.compute_priority_score(detections)
        prompt = self.build_prompt(detections, heuristic)
        raw_response = self.call_llm(prompt)
        parsed_response = self.extract_json(raw_response)

        final_result = {
            "input_detections": detections,
            "heuristic_analysis": heuristic,
            "llm_analysis": parsed_response,
        }
        return final_result


if __name__ == "__main__":
    sample_detections = [
        {
            "class": "car",
            "bbox": [0.51, 0.73, 0.24, 0.20],
            "confidence": 0.93,
        },
        {
            "class": "pedestrian",
            "bbox": [0.47, 0.76, 0.08, 0.17],
            "confidence": 0.89,
        },
        {
            "class": "traffic sign",
            "bbox": [0.84, 0.31, 0.06, 0.11],
            "confidence": 0.78,
        },
    ]

    analyzer = RoadSceneLLMAnalyzer()
    result = analyzer.analyze(sample_detections)
    saved_file = analyzer.save_result(result)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nRésultat sauvegardé dans : {saved_file}")

]
