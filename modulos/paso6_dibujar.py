"""
========================================================
  PASO 6 - Pizarra Digital (Dibujar con el dedo)
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Implementar una pizarra digital donde el usuario
    puede pintar en pantalla arrastrando el dedo índice
    (LM #8). El trazo se superpone sobre el video en
    tiempo real.

  Modo de dibujo:
    · [ESPACIO] o [d] activan/desactivan el modo dibujo
      (toggle — no hace falta mantener pulsado).
    · [b] activa/desactiva el borrador.
    · (El paso 7 reemplazará esto con detección de
      gestos automática.)

  Controles:
    [ESPACIO]  Activar / desactivar modo dibujo (toggle)
    [d]        Activar / desactivar modo dibujo (toggle)
    [b]        Activar / desactivar borrador  (toggle)
    [c]        Cambiar color del pincel
    [+] [-]    Aumentar / reducir grosor del pincel
    [r]        Limpiar lienzo (reset)
    [s]        Captura de pantalla
    [q]        Salir

  Librerías requeridas:
    pip install opencv-python mediapipe
========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import os
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
NOMBRE_VENTANA  = "TouchWall - Paso 6: Pizarra Digital"

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

# Suavizado del trazo
LERP_FACTOR = 0.4       # más alto = más reactivo (menos suavizado)

# Paleta de colores del pincel (BGR)
PALETA_COLORES = [
    (255, 255, 255),    # blanco
    (0,   0,   255),    # rojo
    (0,   165, 255),    # naranja
    (0,   255, 255),    # amarillo
    (0,   255,   0),    # verde
    (255, 0,     0),    # azul
    (255, 0,   255),    # magenta
    (0,   255, 200),    # cian-verde
]
NOMBRES_COLORES = [
    "Blanco", "Rojo", "Naranja", "Amarillo",
    "Verde",  "Azul", "Magenta", "Cian"
]

# Pincel
GROSOR_INICIAL = 8
GROSOR_MIN     = 2
GROSOR_MAX     = 50
GROSOR_PASO    = 2
GROSOR_BORRADOR = 40    # grosor del borrador


# Colores HUD
COLOR_HUD_DRAW  = (0,   220, 100)
COLOR_HUD_IDLE  = (100, 100, 220)
COLOR_HUD_ERASE = (80,  80,  80)
FUENTE = cv2.FONT_HERSHEY_SIMPLEX

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

    print("=" * 54)
    print("  TouchWall — Paso 6: Pizarra Digital")
    print("=" * 54)
    print(f"  Cámara     : {ancho_r}x{alto_r} @ {fps_r:.0f}fps")
    print(f"  Pincel     : grosor={GROSOR_INICIAL}px | {len(PALETA_COLORES)} colores")
    print("=" * 54)
    print("  [ESPACIO] Dibujar (mantener)  [d] Bloquear modo")
    print("  [c] Color  [+/-] Grosor  [b] Borrador  [r] Reset")
    print("  [s] Captura  [q] Salir")
    print("=" * 54)
    return cap


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
            cv2.line(frame, puntos[s], puntos[e], (60, 60, 60), 1)
    for i, (cx, cy) in enumerate(puntos):
        if i != INDICE_LM:
            cv2.circle(frame, (cx, cy), 3, (80, 80, 80), -1)


def dibujar_cursor_indice(frame, cx, cy, dibujando, borrando, color):
    """Cursor del dedo: cambia forma según el modo activo."""
    if borrando:
        cv2.circle(frame, (cx, cy), GROSOR_BORRADOR // 2, (200, 200, 200), 2)
        cv2.putText(frame, "BORRADOR", (cx + 12, cy - 12),
                    FUENTE, 0.45, (200, 200, 200), 1)
    elif dibujando:
        cv2.circle(frame, (cx, cy), 6, color, -1)
        cv2.circle(frame, (cx, cy), 9, (255, 255, 255), 1)
    else:
        cv2.circle(frame, (cx, cy), 10, (180, 180, 180), 1)
        cv2.line(frame, (cx - 14, cy), (cx + 14, cy), (180, 180, 180), 1)
        cv2.line(frame, (cx, cy - 14), (cx, cy + 14), (180, 180, 180), 1)


def dibujar_paleta(frame, idx_color, grosor):
    """Barra de paleta de colores en la parte inferior del frame."""
    alto_f, ancho_f = frame.shape[:2]
    tam    = 32
    margen = 8
    y_base = alto_f - tam - margen

    for i, color in enumerate(PALETA_COLORES):
        x = margen + i * (tam + 4)
        # Fondo negro redondeado
        cv2.rectangle(frame, (x, y_base), (x + tam, y_base + tam), (20, 20, 20), -1)
        # Muestra de color
        cv2.rectangle(frame, (x + 3, y_base + 3),
                      (x + tam - 3, y_base + tam - 3), color, -1)
        # Borde de selección
        if i == idx_color:
            cv2.rectangle(frame, (x, y_base), (x + tam, y_base + tam),
                          (255, 255, 255), 2)

    # Indicador de grosor
    gx = margen + len(PALETA_COLORES) * (tam + 4) + 12
    gy = y_base + tam // 2
    cv2.line(frame, (gx, gy), (gx + 40, gy),
             PALETA_COLORES[idx_color], max(1, grosor // 3))
    cv2.putText(frame, f"{grosor}px", (gx + 46, gy + 5),
                FUENTE, 0.45, (180, 180, 180), 1)


def dibujar_hud(frame, fps, detectado, dibujando, modo_bloqueado,
                borrando, nombre_color, grosor):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    if borrando:
        color_hud = COLOR_HUD_ERASE
        modo_txt  = "BORRADOR  [b]=soltar"
    elif dibujando:
        color_hud = COLOR_HUD_DRAW
        modo_txt  = ("DIBUJANDO [ESPACIO]=soltar"
                     if not modo_bloqueado else "DIBUJANDO [d]=desbloquear")
    else:
        color_hud = COLOR_HUD_IDLE
        modo_txt  = ("[ESPACIO] o [d] para dibujar"
                     if detectado else "Buscando mano...")

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 26), FUENTE, 0.8, color_hud, 2)
    cv2.putText(frame, modo_txt,
                (10, 54), FUENTE, 0.62, color_hud, 1)
    cv2.putText(frame, f"Color: {nombre_color}  Grosor: {grosor}px  "
                        f"[c]=color  [+/-]=grosor  [r]=reset",
                (10, 76), FUENTE, 0.46, (160, 160, 160), 1)

    # LIVE
    alto_f, ancho_f = frame.shape[:2]
    cv2.circle(frame, (ancho_f - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_f - 70, 32),
                FUENTE, 0.55, (0, 0, 255), 1)


def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso6_{ts}.png")
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

    # Lienzo transparente (negro puro = sin trazo)
    lienzo = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)

    # Estado
    idx_color      = 0                          # índice en la paleta
    grosor         = GROSOR_INICIAL
    modo_bloqueado = False                      # toggle dibujo ([ESPACIO]/[d])
    borrando       = False                      # toggle borrador ([b])

    cx_suave = float(ANCHO // 2)
    cy_suave = float(ALTO  // 2)
    cx_prev  = None                             # punto anterior del trazo
    cy_prev  = None

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
        cx_int = cy_int = None

        if resultado.hand_landmarks:
            landmarks = resultado.hand_landmarks[0]
            dibujar_esqueleto_tenue(frame, landmarks, alto_frame, ancho_frame)

            lm = landmarks[INDICE_LM]
            cx_suave = lerp(cx_suave, lm.x * ancho_frame, LERP_FACTOR)
            cy_suave = lerp(cy_suave, lm.y * alto_frame,  LERP_FACTOR)
            cx_int   = int(cx_suave)
            cy_int   = int(cy_suave)
            detectado = True

            # ── Trazar sobre el lienzo ──
            if modo_bloqueado and not borrando:
                color_actual = PALETA_COLORES[idx_color]
                if cx_prev is not None:
                    cv2.line(lienzo, (cx_prev, cy_prev),
                             (cx_int,  cy_int),
                             color_actual, grosor)
                else:
                    cv2.circle(lienzo, (cx_int, cy_int),
                               grosor // 2, color_actual, -1)
                cx_prev, cy_prev = cx_int, cy_int

            elif borrando:
                # Borrar zona circular alrededor del dedo
                cv2.circle(lienzo, (cx_int, cy_int),
                           GROSOR_BORRADOR, (0, 0, 0), -1)
                cx_prev, cy_prev = cx_int, cy_int

            else:
                cx_prev = cy_prev = None   # levantar "lápiz"
        else:
            cx_prev = cy_prev = None       # mano perdida → levantar lápiz

        # ── Combinar lienzo + video ──
        # Máscara: píxeles con trazo (no negros)
        mascara = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(mascara, 1, 255, cv2.THRESH_BINARY)

        # Alpha blend correcto: evita overflow uint8 con np.clip
        # Donde hay trazo: 70% lienzo + 30% video
        ALPHA_LIENZO = 0.75
        frame_final = frame.copy()
        idx = mascara > 0
        frame_final[idx] = np.clip(
            lienzo[idx]  * ALPHA_LIENZO +
            frame[idx]   * (1.0 - ALPHA_LIENZO),
            0, 255
        ).astype(np.uint8)

        # ── Cursor del índice ──
        if detectado and cx_int is not None:
            dibujar_cursor_indice(frame_final, cx_int, cy_int,
                                  modo_bloqueado,
                                  borrando,
                                  PALETA_COLORES[idx_color])

        # ── Paleta de colores ──
        dibujar_paleta(frame_final, idx_color, grosor)

        # ── FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin    = cv2.getTickCount()
            tiempo_seg    = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        # ── HUD ──
        dibujar_hud(frame_final, fps_mostrar, detectado,
                    modo_bloqueado, modo_bloqueado,
                    borrando, NOMBRES_COLORES[idx_color], grosor)

        cv2.imshow(NOMBRE_VENTANA, frame_final)

        # ── Controles (todos son toggles — no "mantener pulsado") ──
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla in (ord(' '), ord('d')):     # ESPACIO o D → toggle dibujo
            modo_bloqueado = not modo_bloqueado
            if not modo_bloqueado:
                cx_prev = cy_prev = None        # levantar lápiz al pausar
            estado = "ACTIVO" if modo_bloqueado else "PAUSADO"
            print(f"  Modo dibujo: {estado}")
        elif tecla == ord('b'):                 # toggle borrador
            borrando = not borrando
            if borrando:
                modo_bloqueado = False          # borrador desactiva dibujo
            cx_prev = cy_prev = None
            print(f"  Borrador: {'ON' if borrando else 'OFF'}")
        elif tecla == ord('c'):                 # siguiente color
            idx_color = (idx_color + 1) % len(PALETA_COLORES)
            print(f"  Color: {NOMBRES_COLORES[idx_color]}")
        elif tecla in (ord('+'), ord('=')):
            grosor = min(grosor + GROSOR_PASO, GROSOR_MAX)
        elif tecla == ord('-'):
            grosor = max(grosor - GROSOR_PASO, GROSOR_MIN)
        elif tecla == ord('r'):
            lienzo[:] = 0
            cx_prev = cy_prev = None
            print("  Lienzo limpiado.")
        elif tecla == ord('s'):
            guardar_captura(frame_final)

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
