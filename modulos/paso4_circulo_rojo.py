"""
========================================================
  PASO 4 - Mover un Círculo Rojo
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Dibujar un círculo rojo en la ventana de OpenCV
    que siga en tiempo real la posición de la punta
    del dedo índice (Landmark #8).

    Características visuales:
      · Círculo rojo principal que sigue al dedo
      · Suavizado de movimiento (lerp) para evitar
        temblor / jitter del sensor
      · Estela de posiciones anteriores (trail)
      · HUD con FPS, posición y estado

  Controles:
    - Presiona 'q' para salir
    - Presiona 's' para tomar una captura de pantalla
    - Presiona '+' / '-' para cambiar el radio del círculo
    - Presiona 'e' para activar / desactivar la estela

  Librerías requeridas:
    pip install opencv-python mediapipe
========================================================
"""

import cv2
import mediapipe as mp
import os
from collections import deque
from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CAMARA_ID       = 0
ANCHO           = 1280
ALTO            = 720
FPS_OBJETIVO    = 30
NOMBRE_VENTANA  = "TouchWall - Paso 4: Círculo Rojo"

# Modelo MediaPipe
RUTA_MODELO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "modelos", "hand_landmarker.task"
)
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Detección
MAX_MANOS               = 1
MIN_CONFIANZA_DETECCION = 0.7
MIN_CONFIANZA_TRACKING  = 0.7
MIN_PRESENCIA           = 0.5
INDICE_LM               = 8     # punta del dedo índice

# Círculo rojo
RADIO_INICIAL   = 28             # radio en píxeles (ajustable con +/-)
RADIO_MIN       = 10
RADIO_MAX       = 80
COLOR_RELLENO   = (0,   0, 220)  # BGR → rojo
COLOR_BORDE     = (0,   0, 160)  # rojo oscuro para el borde
COLOR_BRILLO    = (80, 80, 255)  # toque de brillo (reflexión)

# Suavizado de movimiento (0.0 = sin mover, 1.0 = sin suavizado)
LERP_FACTOR     = 0.25

# Estela
MAX_TRAIL       = 20             # cantidad de posiciones guardadas
COLOR_TRAIL_BASE = (0, 0, 180)   # color base de la estela (rojo oscuro)

# Colores HUD
COLOR_HUD_OK   = (100, 220, 100)
COLOR_HUD_WAIT = (220, 100, 100)
FUENTE         = cv2.FONT_HERSHEY_SIMPLEX

# Conexiones para el esqueleto tenue
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────

def verificar_modelo():
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
    cap = cv2.VideoCapture(CAMARA_ID)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara ID={CAMARA_ID}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS,          FPS_OBJETIVO)
    for _ in range(15):       # warmup
        cap.read()

    ancho_r = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_r  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r   = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 52)
    print("  TouchWall — Paso 4: Círculo Rojo")
    print("=" * 52)
    print(f"  Cámara ID  : {CAMARA_ID}")
    print(f"  Resolución : {ancho_r} x {alto_r} px")
    print(f"  FPS        : {fps_r:.1f}")
    print(f"  Suavizado  : LERP={LERP_FACTOR}")
    print(f"  Estela     : {MAX_TRAIL} posiciones")
    print("=" * 52)
    print("  [q] Salir    [s] Captura")
    print("  [+] Radio+   [-] Radio-   [e] Estela ON/OFF")
    print("=" * 52)
    return cap


# ─────────────────────────────────────────────────────
# LERP (interpolación lineal para suavizado)
# ─────────────────────────────────────────────────────

def lerp(actual, objetivo, factor):
    """Interpolación lineal: mueve 'actual' hacia 'objetivo' suavemente."""
    return actual + (objetivo - actual) * factor


# ─────────────────────────────────────────────────────
# DIBUJO
# ─────────────────────────────────────────────────────

def dibujar_esqueleto_tenue(frame, landmarks, alto, ancho):
    """Esqueleto completo en gris muy tenue como referencia."""
    puntos = [
        (int(lm.x * ancho), int(lm.y * alto))
        for lm in landmarks
    ]
    for con in CONEXIONES_MANO:
        s, e = con.start, con.end
        if s < len(puntos) and e < len(puntos):
            cv2.line(frame, puntos[s], puntos[e], (60, 60, 60), 1)
    for i, (cx, cy) in enumerate(puntos):
        if i != INDICE_LM:
            cv2.circle(frame, (cx, cy), 3, (80, 80, 80), -1)


def dibujar_trail(frame, trail):
    """
    Dibuja la estela de posiciones anteriores del círculo.
    Los puntos más viejos son más pequeños y transparentes.
    """
    n = len(trail)
    for i, (tx, ty) in enumerate(trail):
        ratio   = (i + 1) / n          # 0.0 → más viejo, 1.0 → más reciente
        radio   = max(3, int(10 * ratio))
        alpha   = ratio                 # transparencia proporcional
        # Mezcla color de la estela con el fondo usando addWeighted
        r_canal = int(COLOR_TRAIL_BASE[2] * ratio)  # más rojo al acercarse
        color   = (0, 0, max(60, r_canal))
        cv2.circle(frame, (tx, ty), radio, color, -1)


def dibujar_circulo_rojo(frame, cx, cy, radio):
    """
    Dibuja el círculo rojo principal con efecto de profundidad:
      - Sombra desplazada hacia abajo-derecha
      - Relleno rojo
      - Borde rojo oscuro
      - Punto de brillo en la esquina superior izquierda
    """
    # Sombra
    cv2.circle(frame, (cx + 4, cy + 4), radio, (0, 0, 40), -1)
    # Relleno principal
    cv2.circle(frame, (cx, cy), radio, COLOR_RELLENO, -1)
    # Borde
    cv2.circle(frame, (cx, cy), radio, COLOR_BORDE, 2)
    # Brillo (punto blanco pequeño en la parte superior-izquierda)
    bx = cx - radio // 3
    by = cy - radio // 3
    br = max(3, radio // 5)
    cv2.circle(frame, (bx, by), br, COLOR_BRILLO, -1)


def dibujar_hud(frame, fps, detectado, cx, cy, radio, trail_activo):
    """Panel de información en la esquina superior izquierda."""
    ancho_panel = 400
    alto_panel  = 110 if detectado else 70

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (ancho_panel, alto_panel), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 26), FUENTE, 0.8, COLOR_HUD_OK, 2)

    if detectado:
        cv2.putText(frame, f"Indice  LM#8 → circulo rojo",
                    (10, 54), FUENTE, 0.62, COLOR_HUD_OK, 1)
        trail_str = "ON" if trail_activo else "OFF"
        cv2.putText(frame, f"Pos ({cx}, {cy})  r={radio}  estela={trail_str}",
                    (10, 80), FUENTE, 0.58, COLOR_HUD_OK, 1)
    else:
        cv2.putText(frame, "Buscando mano...",
                    (10, 54), FUENTE, 0.65, COLOR_HUD_WAIT, 2)

    # Indicador LIVE
    alto_f, ancho_f = frame.shape[:2]
    cv2.circle(frame, (ancho_f - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_f - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)

    cv2.putText(frame, "[q]Salir [s]Captura [+/-]Radio [e]Estela",
                (10, alto_f - 12), FUENTE, 0.48, (150, 150, 150), 1)


def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso4_{ts}.png")
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

    # Estado del círculo (posición suavizada con lerp)
    alto_frame, ancho_frame = ALTO, ANCHO
    cx_suave = float(ancho_frame // 2)
    cy_suave = float(alto_frame  // 2)
    radio    = RADIO_INICIAL

    trail        = deque(maxlen=MAX_TRAIL)
    trail_activo = True

    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()
    timestamp_ms    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo leer el frame.")
            break

        frame = cv2.flip(frame, 1)
        alto_frame, ancho_frame = frame.shape[:2]

        # ── MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        resultado = detector.detect_for_video(mp_image, timestamp_ms)

        detectado = False

        if resultado.hand_landmarks:
            landmarks = resultado.hand_landmarks[0]

            # Esqueleto tenue de contexto
            dibujar_esqueleto_tenue(frame, landmarks, alto_frame, ancho_frame)

            # Coordenadas crudas del índice (LM #8)
            lm = landmarks[INDICE_LM]
            cx_raw = lm.x * ancho_frame
            cy_raw = lm.y * alto_frame

            # Suavizado con lerp (reducir jitter)
            cx_suave = lerp(cx_suave, cx_raw, LERP_FACTOR)
            cy_suave = lerp(cy_suave, cy_raw, LERP_FACTOR)

            cx_int = int(cx_suave)
            cy_int = int(cy_suave)

            # Guardar posición en la estela
            if trail_activo:
                trail.append((cx_int, cy_int))

            detectado = True
        else:
            # Sin mano: vaciar estela
            trail.clear()

        # ── Dibujar estela ──
        if trail_activo and trail:
            dibujar_trail(frame, trail)

        # ── Dibujar círculo rojo ──
        if detectado:
            dibujar_circulo_rojo(frame, cx_int, cy_int, radio)

        # ── FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin    = cv2.getTickCount()
            tiempo_seg    = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        # ── HUD ──
        cx_hud = int(cx_suave) if detectado else 0
        cy_hud = int(cy_suave) if detectado else 0
        dibujar_hud(frame, fps_mostrar, detectado, cx_hud, cy_hud,
                    radio, trail_activo)

        # ── Mostrar ──
        cv2.imshow(NOMBRE_VENTANA, frame)

        # ── Controles ──
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla == ord('s'):
            guardar_captura(frame)
        elif tecla == ord('+') or tecla == ord('='):
            radio = min(radio + 4, RADIO_MAX)
            print(f"  Radio: {radio}px")
        elif tecla == ord('-'):
            radio = max(radio - 4, RADIO_MIN)
            print(f"  Radio: {radio}px")
        elif tecla == ord('e'):
            trail_activo = not trail_activo
            trail.clear()
            print(f"  Estela: {'ON' if trail_activo else 'OFF'}")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
