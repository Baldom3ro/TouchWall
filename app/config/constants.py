import os

# ─── RUTAS Y ARCHIVOS ───
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "hand_landmarker.task")
CALIBRATION_FILE = os.path.join(BASE_DIR, "app", "data", "calibration.json")

# ─── CÁMARA ───
CAMARA_ID = 0
ANCHO = 1280
ALTO = 720
FPS_OBJETIVO = 30
NOMBRE_MONITOR = "TouchWall - Monitor"
NOMBRE_CALIB = "TouchWall - Calibracion"

# ─── MEDIAPIPE ───
MAX_MANOS = 2
MIN_CONFIANZA_DETECCION = 0.4
MIN_CONFIANZA_TRACKING = 0.4

# ─── GESTOS (UMBRALES RELATIVOS) ───
UMBRAL_PINCH_INICIO = 0.20  # Aumentado para que sea más fácil hacer clic
UMBRAL_PINCH_FIN    = 0.30

# ─── IDS DE LANDMARKS ───
INDICE_LM = 8
PULGAR_LM = 4
MEDIO_LM = 12
MENIQUE_LM = 20

# ─── CONSTANTES DE GESTOS ───
GESTO_OTRO  = 0
GESTO_PUÑO  = 1
GESTO_PINCH = 2
