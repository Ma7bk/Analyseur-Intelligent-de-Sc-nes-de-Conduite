

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")


MODEL_NAME = "lama-3.3-70b-versatile"


import pathlib
YOLO_MODEL_PATH = str(pathlib.Path(__file__).parent / "best_model_autoroute.pt")
