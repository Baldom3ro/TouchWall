"""
========================================================
  PASO 3 - Detectar Dedo Índice
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    A partir de los 21 landmarks detectados en el
    paso anterior, aislar específicamente el Landmark #8
    (punta del dedo índice) y resaltarlo en pantalla
    con su posición en coordenadas de píxel y
    coordenadas normalizadas (0.0 – 1.0).

  Landmarks relevantes de MediaPipe Hands:
    0  = Muñeca
    4  = Pulgar (punta)
    8  = Índice (punta)   ← este paso
    12 = Medio  (punta)
    16 = Anular (punta)
    20 = Meñique (punta)

  Controles:
    - Presiona 'q' para salir
    - Presiona 's' para tomar una captura de pantalla

  Librerías requeridas:
    pip install opencv-python mediapipe
========================================================
"""

import cv2
import mediapipe as mp
import os
from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CAMARA_ID       = 0       # 0 = webcam real
ANCHO           = 1280
ALTO            = 720
FPS_OBJETIVO    = 30
NOMBRE_VENTANA  = "TouchWall - Paso 3: Detectar Índice"

# Ruta al modelo (reutilizamos el del paso 2)
RUTA_MODELO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "modelos", "hand_landmarker.task"
)
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Parámetros de detección
MAX_MANOS               = 1
MIN_CONFIANZA_DETECCION = 0.7
MIN_CONFIANZA_TRACKING  = 0.7
MIN_PRESENCIA           = 0.5

# Índice del landmark de la punta del dedo índice
INDICE_LM = 8

# Colores (BGR)
COLOR_TODOS_LM      = (100, 100, 100)   # gris tenue: resto de landmarks
COLOR_CONEXIONES    = (80,  80,  80)    # gris oscuro: conexiones
COLOR_INDICE_RELLENO = (0,  220, 255)   # amarillo-cian: círculo interior
COLOR_INDICE_BORDE  = (0,  140, 255)    # naranja: borde del marcador
COLOR_CRUZ          = (255, 255, 255)   # blanco: cruz/crosshair
COLOR_TEXTO         = (0,  220, 255)    # mismo que relleno
COLOR_HUD_OK        = (0,  220, 100)
COLOR_HUD_WAIT      = (0,  100, 220)

FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Conexiones del esqueleto de la mano
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────

def verificar_modelo():
    """Verifica que el modelo existe; si no, lo descarga."""
    if os.path.exists(RUTA_MODELO):
        print(f"  Modelo encontrado: {RUTA_MODELO}")
        return True
    import urllib.request
    print("  Modelo no encontrado. Descargando...")
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    try:
        urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)
        print(f"  [OK] Modelo descargado: {RUTA_MODELO}")
        return True
    except Exception as e:
        print(f"  [ERROR] No se pudo descargar el modelo: {e}")
        return False


def crear_detector():
    """Crea el HandLandmarker con la Tasks API en modo VIDEO."""
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=RUTA_MODELO),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_MANOS,
        min_hand_detection_confidence=MIN_CONFIANZA_DETECCION,
        min_hand_presence_confidence=MIN_PRESENCIA,
        min_tracking_confidence=MIN_CONFIANZA_TRACKING,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def iniciar_camara():
    """Abre la cámara y aplica configuración de resolución/FPS."""
    cap = cv2.VideoCapture(CAMARA_ID)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara ID={CAMARA_ID}")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS,          FPS_OBJETIVO)

    # Warmup: descartar frames iniciales en blanco
    for _ in range(15):
        cap.read()

    ancho_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_real  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real   = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 52)
    print("  TouchWall — Paso 3: Detectar Dedo Índice")
    print("=" * 52)
    print(f"  Cámara ID  : {CAMARA_ID}")
    print(f"  Resolución : {ancho_real} x {alto_real} px")
    print(f"  FPS        : {fps_real:.1f}")
    print(f"  Landmark   : #{INDICE_LM} — punta del dedo índice")
    print("=" * 52)
    print("  [q] Salir    [s] Captura de pantalla")
    print("=" * 52)

    return cap


# ─────────────────────────────────────────────────────
# DIBUJO
# ─────────────────────────────────────────────────────

def dibujar_esqueleto(frame, landmarks, alto, ancho):
    """
    Dibuja el esqueleto completo de la mano en gris tenue
    para dar contexto visual, sin opacar el índice.
    """
    puntos = [
        (int(lm.x * ancho), int(lm.y * alto))
        for lm in landmarks
    ]

    # Conexiones
    for con in CONEXIONES_MANO:
        s, e = con.start, con.end
        if s < len(puntos) and e < len(puntos):
            cv2.line(frame, puntos[s], puntos[e], COLOR_CONEXIONES, 1)

    # Todos los landmarks (pequeños, grises)
    for i, (cx, cy) in enumerate(puntos):
        if i != INDICE_LM:
            cv2.circle(frame, (cx, cy), 4, COLOR_TODOS_LM, -1)

    return puntos


def dibujar_indice(frame, cx, cy):
    """
    Dibuja el marcador destacado sobre la punta del índice:
      - Círculo exterior (borde naranja)
      - Círculo interior (amarillo-cian)
      - Cruz/crosshair blanca
    """
    R_EXTERIOR = 18
    R_INTERIOR = 10
    LARGO_CRUZ = 30

    # Borde exterior
    cv2.circle(frame, (cx, cy), R_EXTERIOR, COLOR_INDICE_BORDE, 2)
    # Relleno interior
    cv2.circle(frame, (cx, cy), R_INTERIOR, COLOR_INDICE_RELLENO, -1)
    # Cruz horizontal
    cv2.line(frame, (cx - LARGO_CRUZ, cy), (cx + LARGO_CRUZ, cy), COLOR_CRUZ, 1)
    # Cruz vertical
    cv2.line(frame, (cx, cy - LARGO_CRUZ), (cx, cy + LARGO_CRUZ), COLOR_CRUZ, 1)


def dibujar_etiqueta_indice(frame, cx, cy, nx, ny, alto, ancho):
    """
    Dibuja la etiqueta flotante con coordenadas de píxel y normalizadas
    junto al marcador del índice.
    """
    linea1 = f"px ({cx}, {cy})"
    linea2 = f"n  ({nx:.3f}, {ny:.3f})"

    # Posición de la caja de texto (ajustar si queda fuera de pantalla)
    tx = cx + 22
    ty = cy - 12
    if tx + 160 > ancho:
        tx = cx - 175
    if ty - 14 < 0:
        ty = cy + 36

    # Fondo semitransparente para la etiqueta
    overlay = frame.copy()
    cv2.rectangle(overlay, (tx - 4, ty - 22), (tx + 160, ty + 22), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.putText(frame, linea1, (tx, ty),      FUENTE, 0.52, COLOR_TEXTO, 1)
    cv2.putText(frame, linea2, (tx, ty + 18), FUENTE, 0.52, COLOR_TEXTO, 1)


def dibujar_hud(frame, fps, detectado, cx=None, cy=None):
    """Dibuja el panel HUD en la esquina superior izquierda."""
    ancho_panel = 380
    alto_panel  = 100 if detectado else 70

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (ancho_panel, alto_panel), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 26), FUENTE, 0.8, COLOR_HUD_OK, 2)

    if detectado:
        cv2.putText(frame, f"Indice detectado  LM#{INDICE_LM}",
                    (10, 54), FUENTE, 0.65, COLOR_HUD_OK, 2)
        cv2.putText(frame, f"Pos: ({cx}, {cy})",
                    (10, 80), FUENTE, 0.6, COLOR_HUD_OK, 1)
    else:
        cv2.putText(frame, "Buscando mano...",
                    (10, 54), FUENTE, 0.65, COLOR_HUD_WAIT, 2)

    # Indicador LIVE
    alto_frame, ancho_frame = frame.shape[:2]
    cv2.circle(frame, (ancho_frame - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_frame - 70, 32),
                FUENTE, 0.55, (0, 0, 255), 1)

    cv2.putText(frame, "[q] Salir  [s] Captura",
                (10, alto_frame - 12), FUENTE, 0.52, (160, 160, 160), 1)


# ─────────────────────────────────────────────────────
# CAPTURA
# ─────────────────────────────────────────────────────

def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso3_{ts}.png")
    cv2.imwrite(ruta, frame)
    print(f"  [OK] Captura guardada: {ruta}")


# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    if not verificar_modelo():
        return

    cap = iniciar_camara()
    if cap is None:
        return

    detector = crear_detector()

    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()
    timestamp_ms    = 0

    ultimo_cx = ultimo_cy = None  # última posición conocida del índice

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo leer el frame.")
            break

        frame = cv2.flip(frame, 1)
        alto, ancho = frame.shape[:2]

        # ── Procesar con MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        resultado = detector.detect_for_video(mp_image, timestamp_ms)

        detectado = False
        cx = cy   = None

        if resultado.hand_landmarks:
            landmarks = resultado.hand_landmarks[0]   # primera mano

            # Dibujar esqueleto tenue
            puntos = dibujar_esqueleto(frame, landmarks, alto, ancho)

            # Extraer coordenadas del índice (LM #8)
            lm_indice = landmarks[INDICE_LM]
            cx = int(lm_indice.x * ancho)
            cy = int(lm_indice.y * alto)
            nx, ny = lm_indice.x, lm_indice.y

            # Dibujar marcador destacado
            dibujar_indice(frame, cx, cy)
            dibujar_etiqueta_indice(frame, cx, cy, nx, ny, alto, ancho)

            detectado = True
            ultimo_cx, ultimo_cy = cx, cy

            # Imprimir en consola (útil para depuración)
            if contador_frames % 10 == 0:
                print(f"  [LM#{INDICE_LM}] px=({cx:4d}, {cy:4d})  "
                      f"norm=({nx:.3f}, {ny:.3f})")

        # ── Calcular FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin    = cv2.getTickCount()
            tiempo_seg    = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        # ── HUD ──
        dibujar_hud(frame, fps_mostrar, detectado, cx, cy)

        # ── Mostrar ──
        cv2.imshow(NOMBRE_VENTANA, frame)

        # ── Controles ──
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla == ord('s'):
            guardar_captura(frame)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
