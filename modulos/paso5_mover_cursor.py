"""
========================================================
  PASO 5 - Mover Cursor del Sistema Operativo
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Usar las coordenadas normalizadas de la punta del
    dedo índice (LM #8) para mover el cursor real del
    sistema operativo con pyautogui.

    El espacio de la cámara (0.0–1.0) se mapea al
    tamaño real de la pantalla. El suavizado LERP
    evita el temblor del sensor.

  Seguridad:
    - Esquina superior-izquierda (0,0) activa el
      failsafe de pyautogui y detiene el control.
    - Tecla 'c' activa / pausa el control del cursor.

  Controles:
    [q]  Salir
    [c]  Activar / pausar control del cursor
    [s]  Captura de pantalla
    [+]  Aumentar suavizado (más lento, más suave)
    [-]  Reducir  suavizado (más rápido, más reactivo)

  Librerías requeridas:
    pip install opencv-python mediapipe pyautogui
========================================================
"""

import cv2
import mediapipe as mp
import pyautogui
import os
from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Deshabilitar el failsafe por movimiento rápido a la esquina
# (lo gestionamos manualmente con la tecla 'c')
pyautogui.FAILSAFE = False
pyautogui.PAUSE    = 0          # sin pausa entre llamadas (máx. velocidad)

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CAMARA_ID    = 0
ANCHO        = 1280
ALTO         = 720
FPS_OBJETIVO = 30
NOMBRE_VENTANA = "TouchWall - Paso 5: Mover Cursor"

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
INDICE_LM               = 8

# Suavizado del cursor (0.0 = sin mover, 1.0 = instantáneo)
# Nivel inicial – ajustable en vivo con + / -
LERP_INICIAL  = 0.15
LERP_MIN      = 0.05
LERP_MAX      = 0.60
LERP_PASO     = 0.05

# Márgenes de la cámara que se recortan del mapeo (evita bordes ruidosos)
# 0.05 = ignorar el 5% de cada borde
MARGEN_H = 0.05
MARGEN_V = 0.05

# Colores HUD (BGR)
COLOR_ACTIVO   = (100, 220,   0)   # verde lima: cursor activo
COLOR_PAUSADO  = (100, 100, 220)   # azul: cursor pausado
COLOR_INDICE   = (0,   220, 255)   # amarillo-cian: marcador del índice
FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# Conexiones del esqueleto
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
    for _ in range(15):
        cap.read()

    ancho_r = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_r  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r   = cap.get(cv2.CAP_PROP_FPS)
    sw, sh  = pyautogui.size()

    print("=" * 54)
    print("  TouchWall — Paso 5: Mover Cursor")
    print("=" * 54)
    print(f"  Cámara     : {ancho_r}x{alto_r} @ {fps_r:.0f}fps")
    print(f"  Pantalla   : {sw}x{sh} px")
    print(f"  Suavizado  : LERP={LERP_INICIAL} (ajustable con +/-)")
    print(f"  Márgenes   : H={MARGEN_H*100:.0f}%  V={MARGEN_V*100:.0f}%")
    print("=" * 54)
    print("  [c] Activar/pausar cursor  [q] Salir")
    print("  [s] Captura  [+] Más suave  [-] Más reactivo")
    print("=" * 54)
    return cap


# ─────────────────────────────────────────────────────
# MAPEO DE COORDENADAS
# ─────────────────────────────────────────────────────

def normalizado_a_pantalla(nx, ny, sw, sh):
    """
    Convierte coordenadas normalizadas (0-1) de la cámara
    a coordenadas de píxel en la pantalla del sistema,
    recortando los márgenes definidos para mayor precisión.

        nx, ny   ∈ [0.0, 1.0]   (espacio de la cámara)
        retorna  (sx, sy)        (píxeles de pantalla)
    """
    # Recortar márgenes (zona activa de la cámara)
    nx_clip = (nx - MARGEN_H) / (1.0 - 2 * MARGEN_H)
    ny_clip = (ny - MARGEN_V) / (1.0 - 2 * MARGEN_V)

    # Invertir X porque el frame está en modo espejo
    # (ya aplicamos cv2.flip(frame, 1), pero las coords normalizadas
    #  de mediapipe ya vienen ajustadas al espejo)
    sx = nx_clip * sw
    sy = ny_clip * sh

    # Clamp para no salir de la pantalla
    sx = max(0, min(sw - 1, sx))
    sy = max(0, min(sh - 1, sy))
    return int(sx), int(sy)


def lerp(actual, objetivo, factor):
    return actual + (objetivo - actual) * factor


# ─────────────────────────────────────────────────────
# DIBUJO
# ─────────────────────────────────────────────────────

def dibujar_esqueleto_tenue(frame, landmarks, alto, ancho):
    puntos = [
        (int(lm.x * ancho), int(lm.y * alto))
        for lm in landmarks
    ]
    for con in CONEXIONES_MANO:
        s, e = con.start, con.end
        if s < len(puntos) and e < len(puntos):
            cv2.line(frame, puntos[s], puntos[e], (55, 55, 55), 1)
    for i, (cx, cy) in enumerate(puntos):
        if i != INDICE_LM:
            cv2.circle(frame, (cx, cy), 3, (75, 75, 75), -1)


def dibujar_marcador_indice(frame, cx, cy, activo):
    """Marcador del índice: color varía según si el cursor está activo."""
    color = COLOR_INDICE if activo else (180, 180, 180)
    cv2.circle(frame, (cx, cy), 18, color, 2)
    cv2.circle(frame, (cx, cy),  8, color, -1)
    cv2.line(frame, (cx - 26, cy), (cx + 26, cy), color, 1)
    cv2.line(frame, (cx, cy - 26), (cx, cy + 26), color, 1)


def dibujar_zona_activa(frame, ancho, alto):
    """
    Dibuja un rectángulo tenue que muestra la zona activa de la cámara
    (excluyendo los márgenes configurados).
    """
    x1 = int(MARGEN_H * ancho)
    y1 = int(MARGEN_V * alto)
    x2 = int((1 - MARGEN_H) * ancho)
    y2 = int((1 - MARGEN_V) * alto)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1)
    cv2.putText(frame, "zona activa", (x1 + 4, y1 + 16),
                FUENTE, 0.42, (80, 80, 80), 1)


def dibujar_hud(frame, fps, detectado, activo, lerp_val,
                cx_cam=None, cy_cam=None, sx=None, sy=None):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (440, 115), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    color = COLOR_ACTIVO if activo else COLOR_PAUSADO
    estado_txt = "CURSOR ACTIVO  [c]=pausar" if activo else "CURSOR PAUSADO [c]=activar"

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 26), FUENTE, 0.8, color, 2)
    cv2.putText(frame, estado_txt,
                (10, 54), FUENTE, 0.6, color, 2)

    if detectado and cx_cam is not None:
        cv2.putText(frame,
                    f"cam ({cx_cam}, {cy_cam})  pantalla ({sx}, {sy})",
                    (10, 80), FUENTE, 0.55, color, 1)
        cv2.putText(frame,
                    f"suavizado LERP={lerp_val:.2f}  [+/-]",
                    (10, 100), FUENTE, 0.5, color, 1)
    else:
        cv2.putText(frame, "Buscando mano...",
                    (10, 80), FUENTE, 0.6, COLOR_PAUSADO, 1)

    # LIVE
    alto_f, ancho_f = frame.shape[:2]
    cv2.circle(frame, (ancho_f - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_f - 70, 32),
                FUENTE, 0.55, (0, 0, 255), 1)
    cv2.putText(frame, "[q]Salir  [c]Cursor  [s]Captura  [+/-]Suavizado",
                (10, alto_f - 12), FUENTE, 0.46, (140, 140, 140), 1)


def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso5_{ts}.png")
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

    detector   = crear_detector()
    sw, sh     = pyautogui.size()       # resolución de pantalla

    # Estado
    cursor_activo  = False              # empieza pausado (seguro)
    lerp_val       = LERP_INICIAL

    # Posición suavizada del cursor en pantalla
    sx_suave = float(sw // 2)
    sy_suave = float(sh // 2)

    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()
    timestamp_ms    = 0

    print("  Control del cursor PAUSADO — presiona [c] para activar")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo leer el frame.")
            break

        frame = cv2.flip(frame, 1)
        alto_frame, ancho_frame = frame.shape[:2]

        # Dibujar zona activa de mapeo
        dibujar_zona_activa(frame, ancho_frame, alto_frame)

        # ── MediaPipe ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        resultado = detector.detect_for_video(mp_image, timestamp_ms)

        detectado = False
        cx_cam = cy_cam = sx_final = sy_final = None

        if resultado.hand_landmarks:
            landmarks = resultado.hand_landmarks[0]
            dibujar_esqueleto_tenue(frame, landmarks, alto_frame, ancho_frame)

            lm = landmarks[INDICE_LM]
            nx, ny = lm.x, lm.y

            # Coordenadas en la cámara (para HUD)
            cx_cam = int(nx * ancho_frame)
            cy_cam = int(ny * alto_frame)

            # Mapeo a pantalla
            sx_raw, sy_raw = normalizado_a_pantalla(nx, ny, sw, sh)

            # Suavizado en espacio de pantalla
            sx_suave = lerp(sx_suave, sx_raw, lerp_val)
            sy_suave = lerp(sy_suave, sy_raw, lerp_val)
            sx_final = int(sx_suave)
            sy_final = int(sy_suave)

            # Mover cursor real
            if cursor_activo:
                pyautogui.moveTo(sx_final, sy_final)

            # Marcador visual en el frame
            dibujar_marcador_indice(frame, cx_cam, cy_cam, cursor_activo)
            detectado = True

        # ── FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin    = cv2.getTickCount()
            tiempo_seg    = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        # ── HUD ──
        dibujar_hud(frame, fps_mostrar, detectado, cursor_activo,
                    lerp_val, cx_cam, cy_cam, sx_final, sy_final)

        cv2.imshow(NOMBRE_VENTANA, frame)

        # ── Controles ──
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla == ord('c'):
            cursor_activo = not cursor_activo
            estado = "ACTIVO ✓" if cursor_activo else "PAUSADO ✗"
            print(f"  Control cursor: {estado}")
        elif tecla == ord('s'):
            guardar_captura(frame)
        elif tecla in (ord('+'), ord('=')):
            lerp_val = min(lerp_val + LERP_PASO, LERP_MAX)
            print(f"  LERP={lerp_val:.2f}  (más reactivo)")
        elif tecla == ord('-'):
            lerp_val = max(lerp_val - LERP_PASO, LERP_MIN)
            print(f"  LERP={lerp_val:.2f}  (más suave)")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
