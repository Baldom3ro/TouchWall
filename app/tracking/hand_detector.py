import mediapipe as mp
import cv2
from app.config.constants import *

class HandDetector:
    """
    Envoltura (Wrapper) para MediaPipe Hand Landmarker.
    Maneja la inicialización del modelo de IA y la detección de manos.
    """
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=MAX_MANOS,
            min_hand_detection_confidence=MIN_CONFIANZA_DETECCION,
            min_hand_presence_confidence=MIN_CONFIANZA_TRACKING,
            min_tracking_confidence=MIN_CONFIANZA_TRACKING
        )
        
        try:
            self.detector = HandLandmarker.create_from_options(options)
            print("[Info] MediaPipe HandLandmarker cargado correctamente.")
        except Exception as e:
            print(f"[Error] No se pudo cargar el modelo MediaPipe: {e}")
            self.detector = None

    def detect(self, frame_rgb, timestamp_ms):
        """
        Recibe un fotograma en RGB y su marca de tiempo, 
        y devuelve los resultados de la Inteligencia Artificial.
        """
        if self.detector is None or frame_rgb is None:
            return None
            
        # Convertir a formato nativo de MediaPipe
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Procesar
        res = self.detector.detect_for_video(mp_img, timestamp_ms)
        return res

    def close(self):
        """Libera los recursos de MediaPipe."""
        if self.detector is not None:
            self.detector.close()
