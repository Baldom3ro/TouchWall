"""
========================================================
  PASO 2 - Detectar Mano
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Usar MediaPipe HandLandmarker (nueva Tasks API)
    para detectar la mano en el flujo de video y
    dibujar los 21 landmarks (puntos clave) en
    tiempo real.

  Controles:
    - Presiona 'q' para salir
    - Presiona 's' para tomar una captura de pantalla

  Librerías requeridas:
    pip install opencv-python mediapipe
  
  Modelo requerido (descargado automáticamente en setup):
    modelos/hand_landmarker.task
========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CAMARA_ID       = 0       # 0 = webcam real (640x480) | 1 = dispositivo virtual (negro)
ANCHO           = 1280
ALTO            = 720
FPS_OBJETIVO    = 30
NOMBRE_VENTANA  = "TouchWall - Paso 2: Detectar Mano"

# Ruta al modelo de detección
RUTA_MODELO     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelos", "hand_landmarker.task")
URL_MODELO      = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Parámetros de MediaPipe
MAX_MANOS               = 1
MIN_CONFIANZA_DETECCION = 0.7
MIN_CONFIANZA_TRACKING  = 0.7
MIN_PRESENCIA           = 0.5

# Colores (BGR)
COLOR_LANDMARKS    = (0, 255, 0)
COLOR_CONEXIONES   = (255, 255, 255)
COLOR_INFO         = (0, 255, 0)
COLOR_DETECTADO    = (0, 255, 100)
COLOR_NO_DETECTADO = (0, 100, 220)

FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Conexiones de los landmarks de la mano (21 puntos) — Tasks API
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# VERIFICAR / DESCARGAR MODELO
# ─────────────────────────────────────────────────────

def verificar_modelo():
    """Verifica que el modelo existe, si no lo descarga automáticamente."""
    if os.path.exists(RUTA_MODELO):
        print(f"  Modelo encontrado: {RUTA_MODELO}")
        return True

    print(f"  Modelo no encontrado. Descargando...")
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    try:
        urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)
        print(f"  [OK] Modelo descargado: {RUTA_MODELO}")
        return True
    except Exception as e:
        print(f"  [ERROR] No se pudo descargar el modelo: {e}")
        return False


# ─────────────────────────────────────────────────────
# INICIALIZAR MEDIAPIPE (Nueva Tasks API)
# ─────────────────────────────────────────────────────

def crear_detector_manos():
    """
    Crea el HandLandmarker usando la nueva MediaPipe Tasks API.
    Modo VIDEO: procesa frames secuenciales con timestamp.
    """
    BaseOptions = mp_python.BaseOptions
    HandLandmarker = mp_vision.HandLandmarker
    HandLandmarkerOptions = mp_vision.HandLandmarkerOptions
    RunningMode = mp_vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=RUTA_MODELO),
        running_mode=RunningMode.VIDEO,
        num_hands=MAX_MANOS,
        min_hand_detection_confidence=MIN_CONFIANZA_DETECCION,
        min_hand_presence_confidence=MIN_PRESENCIA,
        min_tracking_confidence=MIN_CONFIANZA_TRACKING,
    )

    detector = HandLandmarker.create_from_options(options)
    return detector


# ─────────────────────────────────────────────────────
# INICIALIZAR CÁMARA
# ─────────────────────────────────────────────────────

def iniciar_camara():
    """Abre la cámara y configura su resolución y FPS."""
    cap = cv2.VideoCapture(CAMARA_ID)  # backend por defecto (MSMF en Windows)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara ID={CAMARA_ID}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS,          FPS_OBJETIVO)

    # Warmup: descartar frames iniciales en blanco (Windows necesita ~10 frames)
    for _ in range(15):
        cap.read()

    ancho_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_real  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real   = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 50)
    print("  TouchWall — Paso 2: Detectar Mano")
    print("=" * 50)
    print(f"  Cámara ID  : {CAMARA_ID}")
    print(f"  Resolución : {ancho_real} x {alto_real} px")
    print(f"  FPS        : {fps_real}")
    print(f"  Max manos  : {MAX_MANOS}")
    print(f"  Confianza  : detección={MIN_CONFIANZA_DETECCION} | tracking={MIN_CONFIANZA_TRACKING}")
    print("=" * 50)
    print("  [q] Salir    [s] Captura de pantalla")
    print("=" * 50)

    return cap


# ─────────────────────────────────────────────────────
# DIBUJAR LANDMARKS DE LA MANO
# ─────────────────────────────────────────────────────

def dibujar_mano(frame, resultado):
    """
    Dibuja los 21 landmarks y sus conexiones sobre el frame.
    Retorna True si se detectó al menos una mano.
    """
    if not resultado.hand_landmarks:
        return False

    alto, ancho = frame.shape[:2]

    for hand_landmarks in resultado.hand_landmarks:
        # Convertir landmarks normalizados (0-1) a píxeles
        puntos = []
        for lm in hand_landmarks:
            cx = int(lm.x * ancho)
            cy = int(lm.y * alto)
            puntos.append((cx, cy))

        # ── Dibujar conexiones (líneas entre landmarks) ──
        for conexion in CONEXIONES_MANO:
            inicio = conexion.start
            fin    = conexion.end
            if inicio < len(puntos) and fin < len(puntos):
                cv2.line(frame, puntos[inicio], puntos[fin], COLOR_CONEXIONES, 2)

        # ── Dibujar puntos (landmarks) ──
        for i, (cx, cy) in enumerate(puntos):
            # Puntas de dedos (landmarks 4, 8, 12, 16, 20) más grandes
            radio = 8 if i in [4, 8, 12, 16, 20] else 5
            cv2.circle(frame, (cx, cy), radio, COLOR_LANDMARKS, -1)
            cv2.circle(frame, (cx, cy), radio + 2, COLOR_CONEXIONES, 1)

    return True


# ─────────────────────────────────────────────────────
# DIBUJAR HUD
# ─────────────────────────────────────────────────────

def dibujar_hud(frame, fps_actual, mano_detectada):
    """Dibuja el HUD con FPS y estado de detección."""
    alto, ancho = frame.shape[:2]

    # Panel semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"FPS: {fps_actual:.1f}",
                (10, 28), FUENTE, 0.8, COLOR_INFO, 2)

    if mano_detectada:
        txt   = "Mano detectada  (21 landmarks)"
        color = COLOR_DETECTADO
    else:
        txt   = "Buscando mano..."
        color = COLOR_NO_DETECTADO

    cv2.putText(frame, txt,
                (10, 58), FUENTE, 0.65, color, 2)

    cv2.putText(frame, "[q] Salir  [s] Captura",
                (10, alto - 15), FUENTE, 0.55, (180, 180, 180), 1)

    # Indicador LIVE
    cv2.circle(frame, (ancho - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)


# ─────────────────────────────────────────────────────
# GUARDAR CAPTURA
# ─────────────────────────────────────────────────────

def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = os.path.join(carpeta, f"captura_{timestamp}.png")
    cv2.imwrite(nombre, frame)
    print(f"  [OK] Captura guardada: {nombre}")


# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    if not verificar_modelo():
        return

    cap = iniciar_camara()
    if cap is None:
        return

    detector = crear_detector_manos()  # Tasks API — retorna HandLandmarker

    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()
    timestamp_ms    = 0  # timestamp requerido por la API de video

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo leer el frame.")
            break

        # ── Voltear horizontalmente (efecto espejo) ──
        frame = cv2.flip(frame, 1)

        # ── Convertir a RGB para MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Crear imagen de MediaPipe y procesar ──
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33  # ~30 FPS → ~33ms por frame
        resultado = detector.detect_for_video(mp_image, timestamp_ms)

        # ── Dibujar landmarks ──
        mano_detectada = dibujar_mano(frame, resultado)

        # ── Calcular FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin    = cv2.getTickCount()
            tiempo_seg    = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        # ── HUD ──
        dibujar_hud(frame, fps_mostrar, mano_detectada)

        # ── Mostrar ──
        cv2.imshow(NOMBRE_VENTANA, frame)

        # ── Controles ──
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla == ord('s'):
            guardar_captura(frame)

    # ── Liberar recursos ──
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
