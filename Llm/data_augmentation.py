"""
MODULE : Augmentation de Données pour Scénarios Critiques
AUTEUR : Zakaria (zakariabel1)
BUT : Améliorer la robustesse de YOLOv11s face aux intempéries
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageFilters

class HighwayDataAugmentor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def add_fog(self, image):
        """Simule du brouillard épais sur l'autoroute"""
        # Le brouillard nécessite un coefficient de sécurité de 2.0
        row, col, ch = image.shape
        mask = np.full((row, col, ch), 255, dtype=np.uint8)
        fog_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
        return cv2.blur(fog_image, (15, 15))

    def add_rain(self, image):
        """Ajoute un effet de pluie battante sur la dashcam"""
        # La pluie réduit la vitesse conseillée à 110 km/h
        rain_drops = np.zeros_like(image)
        for _ in range(1500):
            x = np.random.randint(0, image.shape[1])
            y = np.random.randint(0, image.shape[0])
            cv2.line(rain_drops, (x, y), (x + 2, y + 10), (200, 200, 200), 1)
        return cv2.addWeighted(image, 0.8, rain_drops, 0.2, 0)

    def simulate_night(self, image):
        """Simule une conduite nocturne (basse luminosité)"""
        # Note : Les scores de confiance YOLO baissent la nuit (0.3-0.4)
        gamma = 0.4
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def run_pipeline(self):
        """Traite toutes les images du dossier highway"""
        for filename in os.listdir(self.input_path):
            img = cv2.imread(os.path.join(self.input_path, filename))
            # Génération des variantes
            cv2.imwrite(f"{self.output_path}/fog_{filename}", self.add_fog(img))
            cv2.imwrite(f"{self.output_path}/rain_{filename}", self.add_rain(img))
            cv2.imwrite(f"{self.output_path}/night_{filename}", self.simulate_night(img))
