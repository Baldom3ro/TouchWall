"""
========================================================
  PASO 7 - Gestos Intuitivos (v2)
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================

  Mapa de gestos
  ══════════════════════════════════════════════════
  👆  ÍNDICE    (solo índice extendido)
        → Modo PUNTERO: mueve el cursor, no dibuja

  🤏  PINCH     (pulgar + índice juntos)
        → DIBUJAR mientras se mantiene el gesto

  ✌️  VICTORIA  (índice + medio extendidos)
        → Cambiar al SIGUIENTE COLOR (disparo único)

  ✋  PALMA     (4 dedos extendidos)
        → BORRADOR activo mientras se mantiene

  ✊  PUÑO      (todos los dedos cerrados)
        → PAUSAR: detiene todo sin cerrar

  🤟  TRES      (índice + medio + anular)
        → LIMPIAR el lienzo (disparo único)

  👍  PULGAR    (solo pulgar extendido)
        → CAPTURA de pantalla (disparo único)
  ══════════════════════════════════════════════════

  Lógica de estado:
    · "Modo continuo" (PINCH/PALMA/PUÑO): activo
      mientras se mantiene el gesto.
    · "Disparo único" (VICTORIA/TRES/PULGAR): se
      ejecuta una vez al confirmar el gesto.

  Teclado (respaldo):
    [q] Salir  [s] Captura  [r] Limpiar  [c] Color
    [+/-] Grosor

  Librerías requeridas:
    pip install opencv-python mediapipe
========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import os
from datetime import datetime
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CAMARA_ID      = 0
ANCHO          = 1280
ALTO           = 720
FPS_OBJETIVO   = 30
NOMBRE_VENTANA = "TouchWall - Paso 7: Gestos Intuitivos"

RUTA_MODELO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "modelos", "hand_landmarker.task"
)
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MAX_MANOS               = 1
MIN_CONFIANZA_DETECCION = 0.7
MIN_CONFIANZA_TRACKING  = 0.7
MIN_PRESENCIA           = 0.5
INDICE_LM               = 8

# Suavizado del movimiento del dedo
LERP_FACTOR = 0.35

# Umbrales de detección
UMBRAL_PINCH       = 0.055   # distancia pulgar-índice para PINCH

# Estabilizador: frames consecutivos para confirmar un gesto
FRAMES_ESTABLE     = 5       # ~167ms a 30fps

# Pizarra
PALETA_COLORES = [
    (255, 255, 255),  # blanco
    (0,   0,   255),  # rojo
    (0,   165, 255),  # naranja
    (0,   255, 255),  # amarillo
    (0,   255,   0),  # verde
    (255, 0,     0),  # azul
    (255, 0,   255),  # magenta
    (0,   255, 200),  # cian
]
NOMBRES_COLORES = [
    "Blanco","Rojo","Naranja","Amarillo","Verde","Azul","Magenta","Cian"
]

GROSOR_INICIAL  = 10
GROSOR_MIN      = 2
GROSOR_MAX      = 50
GROSOR_BORRADOR = 45
ALPHA_LIENZO    = 0.80

# Gestos reconocidos
GESTO_INDICE   = "INDICE"
GESTO_PINCH    = "PINCH"
GESTO_VICTORIA = "VICTORIA"
GESTO_PALMA    = "PALMA"
GESTO_PUÑO     = "PUÑO"
GESTO_TRES     = "TRES"
GESTO_PULGAR   = "PULGAR"
GESTO_OTRO     = "OTRO"

# Datos visuales de cada gesto (emoji, descripción, color BGR)
META_GESTOS = {
    GESTO_INDICE:   ("☝",  "PUNTERO",        (0,   220, 255)),
    GESTO_PINCH:    ("🤏", "DIBUJAR",         (0,   180, 80)),
    GESTO_VICTORIA: ("✌",  "CAMBIAR COLOR",   (200, 100, 255)),
    GESTO_PALMA:    ("✋", "BORRADOR",         (180, 180, 180)),
    GESTO_PUÑO:     ("✊", "PAUSA",            (80,   80,  80)),
    GESTO_TRES:     ("🤟", "LIMPIAR LIENZO",  (60,  160, 220)),
    GESTO_PULGAR:   ("👍", "CAPTURA",          (0,   220, 200)),
    GESTO_OTRO:     ("…",  "—",               (100, 100, 100)),
}

FUENTE          = cv2.FONT_HERSHEY_SIMPLEX
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# DETECCIÓN DE GESTOS
# ─────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _dedo_extendido(lm, punta, pip):
    """True si la punta del dedo está más lejos de la muñeca que su articulación intermedia (PIP)."""
    muneca = lm[0]
    return dist(lm[punta], muneca) > dist(lm[pip], muneca)


def _pulgar_extendido(lm):
    """Para el pulgar, comparamos la punta (4) con el nudillo (2)."""
    muneca = lm[0]
    return dist(lm[4], muneca) > dist(lm[2], muneca)


def clasificar_gesto(lm) -> str:
    """
    Clasifica el gesto actual de la mano a partir de los 21 landmarks.
    Retorna uno de los GESTO_* definidos.
    """
    ext_pulgar  = _pulgar_extendido(lm)
    ext_indice  = _dedo_extendido(lm, 8,  6)
    ext_medio   = _dedo_extendido(lm, 12, 10)
    ext_anular  = _dedo_extendido(lm, 16, 14)
    ext_menique = _dedo_extendido(lm, 20, 18)
    d_pinch     = dist(lm[4], lm[8])

    # ── Pinch: pulgar + índice muy juntos ──
    if d_pinch < UMBRAL_PINCH:
        return GESTO_PINCH

    # ── Puño: todos los dedos (sin pulgar) cerrados ──
    if (not ext_indice and not ext_medio
            and not ext_anular and not ext_menique):
        return GESTO_PUÑO

    # ── Palma: todos los dedos extendidos ──
    if ext_indice and ext_medio and ext_anular and ext_menique:
        return GESTO_PALMA

    # ── Tres dedos: índice + medio + anular ──
    if ext_indice and ext_medio and ext_anular and not ext_menique:
        return GESTO_TRES

    # ── Victoria: solo índice + medio ──
    if ext_indice and ext_medio and not ext_anular and not ext_menique:
        return GESTO_VICTORIA

    # ── Solo índice ──
    if ext_indice and not ext_medio and not ext_anular and not ext_menique:
        return GESTO_INDICE

    # ── Pulgar solo ──
    if ext_pulgar and not ext_indice and not ext_medio:
        return GESTO_PULGAR

    return GESTO_OTRO


# ─────────────────────────────────────────────────────
# ESTABILIZADOR DE GESTOS
# ─────────────────────────────────────────────────────

class Estabilizador:
    """
    Confirma un gesto solo cuando se mantiene N frames consecutivos.
    Distingue entre gestos "continuos" (activos mientras se mantienen)
    y gestos "disparo" (se ejecutan una sola vez al confirmarse).
    """
    # Gestos que se activan solo una vez al confirmarse
    GESTOS_DISPARO = {GESTO_VICTORIA, GESTO_TRES, GESTO_PULGAR}

    def __init__(self, n=FRAMES_ESTABLE):
        self.n              = n
        self.gesto_bruto    = GESTO_OTRO
        self.contador       = 0
        self.confirmado     = GESTO_OTRO
        self.confirmado_prev = GESTO_OTRO
        self.disparo        = False   # True solo el frame de activación

    def actualizar(self, gesto_nuevo: str) -> str:
        self.disparo = False

        if gesto_nuevo == self.gesto_bruto:
            self.contador = min(self.contador + 1, self.n)
        else:
            self.gesto_bruto = gesto_nuevo
            self.contador    = 1

        # Gesto confirmado cuando lleva N frames consecutivos
        if self.contador >= self.n:
            nuevo_confirmado = self.gesto_bruto
        else:
            nuevo_confirmado = self.confirmado  # mantener el anterior

        # Detectar cambio de gesto
        if nuevo_confirmado != self.confirmado:
            self.confirmado_prev = self.confirmado
            self.confirmado      = nuevo_confirmado
            if nuevo_confirmado in self.GESTOS_DISPARO:
                self.disparo = True

        return self.confirmado

    @property
    def progreso(self):
        return min(self.contador / self.n, 1.0)


# ─────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────

def verificar_modelo():
    if os.path.exists(RUTA_MODELO):
        print(f"  Modelo encontrado: {RUTA_MODELO}")
        return True
    import urllib.request
    print("  Descargando modelo...")
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    try:
        urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)
        print(f"  [OK] Modelo descargado.")
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
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
        print(f"[ERROR] Cámara {CAMARA_ID} no disponible")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS,          FPS_OBJETIVO)
    for _ in range(15):
        cap.read()

    aw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("=" * 56)
    print("  TouchWall — Paso 7: Gestos Intuitivos")
    print("=" * 56)
    print(f"  Cámara  : {aw}x{ah} @ {fps:.0f}fps")
    print("=" * 56)
    print("  ☝  ÍNDICE    → puntero (sin dibujar)")
    print("  🤏 PINCH     → dibujar (mientras se mantiene)")
    print("  ✌  VICTORIA  → siguiente color")
    print("  ✋ PALMA      → borrador (mientras se mantiene)")
    print("  ✊ PUÑO       → pausa")
    print("  🤟 3 DEDOS   → limpiar lienzo")
    print("  👍 PULGAR    → captura de pantalla")
    print("=" * 56)
    return cap


def lerp(actual, objetivo, factor):
    return actual + (objetivo - actual) * factor


def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso7_{ts}.png")
    cv2.imwrite(ruta, frame)
    print(f"  [OK] Captura guardada: {ruta}")
    return ruta


# ─────────────────────────────────────────────────────
# DIBUJO / UI
# ─────────────────────────────────────────────────────

def dibujar_esqueleto(frame, lm, alto, ancho, gesto):
    """Esqueleto de la mano con color según el gesto activo."""
    _, _, color = META_GESTOS.get(gesto, META_GESTOS[GESTO_OTRO])
    # Versión atenuada para las conexiones
    color_con = tuple(max(0, c // 2) for c in color)

    puntos = [(int(l.x * ancho), int(l.y * alto)) for l in lm]
    for con in CONEXIONES_MANO:
        s, e = con.start, con.end
        if s < len(puntos) and e < len(puntos):
            cv2.line(frame, puntos[s], puntos[e], color_con, 1)
    for i, (cx, cy) in enumerate(puntos):
        r = 6 if i in [4, 8, 12, 16, 20] else 3
        cv2.circle(frame, (cx, cy), r, color, -1)


def dibujar_cursor(frame, cx, cy, gesto, color_pincel):
    """Cursor visual sobre la punta del índice, adaptado al gesto."""
    emoji_map = {
        GESTO_INDICE:   (COLOR_PUNTERO := (0, 220, 255), 14, False),
        GESTO_PINCH:    (color_pincel,                    18, True ),
        GESTO_PALMA:    ((180, 180, 180),                 22, False),
        GESTO_PUÑO:     ((80,   80,  80),                 12, False),
        GESTO_VICTORIA: ((200, 100, 255),                 14, False),
        GESTO_TRES:     ((60,  160, 220),                 16, False),
        GESTO_PULGAR:   ((0,   220, 200),                 14, False),
        GESTO_OTRO:     ((120, 120, 120),                 10, False),
    }
    color, radio, relleno = emoji_map.get(gesto, emoji_map[GESTO_OTRO])

    if relleno:
        cv2.circle(frame, (cx, cy), radio, color, -1)
        cv2.circle(frame, (cx, cy), radio + 4, color, 2)
    else:
        cv2.circle(frame, (cx, cy), radio, color, 2)
        cv2.circle(frame, (cx, cy),  5, color, -1)

    # Líneas de mira
    L = radio + 12
    cv2.line(frame, (cx - L, cy), (cx + L, cy), color, 1)
    cv2.line(frame, (cx, cy - L), (cx, cy + L), color, 1)


def dibujar_leyenda_gestos(frame, gesto_actual, progreso):
    """
    Panel lateral derecho con la leyenda de todos los gestos.
    El gesto activo se resalta.
    """
    alto_f, ancho_f = frame.shape[:2]
    panel_w = 230
    px      = ancho_f - panel_w - 8
    py      = 8
    fila_h  = 30
    n       = len(META_GESTOS)
    panel_h = n * fila_h + 24

    overlay = frame.copy()
    cv2.rectangle(overlay, (px - 4, py), (px + panel_w, py + panel_h),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "GESTOS", (px, py + 14),
                FUENTE, 0.52, (160, 160, 160), 1)

    for i, (gesto_key, (emoji, desc, color)) in enumerate(META_GESTOS.items()):
        if gesto_key == GESTO_OTRO:
            continue
        fy = py + 24 + i * fila_h
        activo = (gesto_key == gesto_actual)

        # Fondo resaltado si es el gesto activo
        if activo:
            cv2.rectangle(frame, (px - 2, fy - 2),
                          (px + panel_w, fy + fila_h - 6),
                          tuple(c // 4 for c in color), -1)
            # Barra de confirmación a la izquierda
            bh = fila_h - 8
            fill = int(bh * progreso)
            cv2.rectangle(frame, (px - 6, fy + bh - fill),
                          (px - 2, fy + bh), color, -1)

        c_texto = color if activo else (110, 110, 110)
        label   = f"{emoji}  {desc}"
        cv2.putText(frame, label, (px + 4, fy + fila_h - 12),
                    FUENTE, 0.48, c_texto,
                    2 if activo else 1, cv2.LINE_AA)


def dibujar_hud(frame, fps, gesto, nombre_color, grosor, pausado):
    """Panel superior izquierdo con estado actual."""
    _, _, color = META_GESTOS.get(gesto, META_GESTOS[GESTO_OTRO])

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    estado = "PAUSADO ✊" if pausado else f"{gesto}"
    cv2.putText(frame, f"FPS: {fps:.1f}   Gesto: {estado}",
                (10, 26), FUENTE, 0.75, color, 2)
    cv2.putText(frame, f"Color: {nombre_color}  Grosor: {grosor}px  "
                        f"  [q]salir [s]captura [r]limpiar [c]color [+/-]grosor",
                (10, 54), FUENTE, 0.44, (140, 140, 140), 1)

    # LIVE
    alto_f, ancho_f = frame.shape[:2]
    cv2.circle(frame, (ancho_f - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_f - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)


def dibujar_feedback_disparo(frame, mensaje, color):
    """Flash de confirmación temporal en pantalla al ejecutar un disparo."""
    alto_f, ancho_f = frame.shape[:2]
    ts = cv2.getTextSize(mensaje, FUENTE, 1.0, 2)[0]
    cx = (ancho_f - ts[0]) // 2
    cy = alto_f // 2
    # Sombra
    cv2.putText(frame, mensaje, (cx + 2, cy + 2), FUENTE, 1.0, (0, 0, 0), 4)
    # Texto
    cv2.putText(frame, mensaje, (cx, cy), FUENTE, 1.0, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    if not verificar_modelo():
        return

    cap = iniciar_camara()
    if cap is None:
        return

    detector  = crear_detector()
    estabiliz = Estabilizador()

    # Estado de la pizarra
    lienzo    = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    idx_color = 0
    grosor    = GROSOR_INICIAL
    cx_prev   = cy_prev = None

    cx_suave  = float(ANCHO // 2)
    cy_suave  = float(ALTO  // 2)

    # Feedback visual temporal (mensaje + duración en frames)
    feedback_msg   = ""
    feedback_color = (255, 255, 255)
    feedback_frames = 0

    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()
    timestamp_ms    = 0

    print("  Listo — muestra tu mano y usa los gestos del panel.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AVISO] No se pudo leer el frame.")
            break

        frame = cv2.flip(frame, 1)
        alto_frame, ancho_frame = frame.shape[:2]

        # ── MediaPipe ──
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms += 33
        resultado = detector.detect_for_video(mp_image, timestamp_ms)

        gesto_confirmado = GESTO_OTRO
        cx_int = cy_int = None
        detectado = False

        if resultado.hand_landmarks:
            lm = resultado.hand_landmarks[0]
            detectado = True

            # Posición suavizada del índice (LM 8)
            cx_suave = lerp(cx_suave, lm[INDICE_LM].x * ancho_frame, LERP_FACTOR)
            cy_suave = lerp(cy_suave, lm[INDICE_LM].y * alto_frame,  LERP_FACTOR)
            cx_int   = int(cx_suave)
            cy_int   = int(cy_suave)

            # Clasificar y estabilizar gesto
            gesto_raw        = clasificar_gesto(lm)
            gesto_confirmado = estabiliz.actualizar(gesto_raw)

            # ── Reaccionar a gestos de DISPARO (uno-vez) ──
            if estabiliz.disparo:
                if gesto_confirmado == GESTO_VICTORIA:
                    idx_color = (idx_color + 1) % len(PALETA_COLORES)
                    msg = f"Color: {NOMBRES_COLORES[idx_color]}"
                    print(f"  [✌] {msg}")
                    feedback_msg    = msg
                    feedback_color  = PALETA_COLORES[idx_color]
                    feedback_frames = 45

                elif gesto_confirmado == GESTO_TRES:
                    lienzo[:] = 0
                    cx_prev = cy_prev = None
                    print("  [🤟] Lienzo limpiado.")
                    feedback_msg    = "Lienzo limpiado"
                    feedback_color  = (60, 160, 220)
                    feedback_frames = 40

                elif gesto_confirmado == GESTO_PULGAR:
                    ruta = guardar_captura(frame)
                    feedback_msg    = "Captura guardada!"
                    feedback_color  = (0, 220, 200)
                    feedback_frames = 50

            # ── Modos CONTINUOS (activos mientras se mantiene el gesto) ──
            pausado = (gesto_confirmado == GESTO_PUÑO)

            if not pausado:
                dibujando = (gesto_confirmado == GESTO_PINCH)
                borrando  = (gesto_confirmado == GESTO_PALMA)

                if dibujando and cx_int is not None:
                    color_actual = PALETA_COLORES[idx_color]
                    if cx_prev is not None:
                        cv2.line(lienzo, (cx_prev, cy_prev),
                                 (cx_int, cy_int), color_actual, grosor)
                    else:
                        cv2.circle(lienzo, (cx_int, cy_int),
                                   grosor // 2, color_actual, -1)
                    cx_prev, cy_prev = cx_int, cy_int

                elif borrando and cx_int is not None:
                    cv2.circle(lienzo, (cx_int, cy_int),
                               GROSOR_BORRADOR, (0, 0, 0), -1)
                    cx_prev, cy_prev = cx_int, cy_int

                else:
                    cx_prev = cy_prev = None
            else:
                cx_prev = cy_prev = None

            # Esqueleto
            dibujar_esqueleto(frame, lm, alto_frame, ancho_frame,
                              gesto_confirmado)
        else:
            estabiliz.actualizar(GESTO_OTRO)
            cx_prev = cy_prev = None
            pausado = False

        # ── Componer lienzo sobre video ──
        mascara = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(mascara, 1, 255, cv2.THRESH_BINARY)
        frame_final = frame.copy()
        idx_mask = mascara > 0
        frame_final[idx_mask] = np.clip(
            lienzo[idx_mask] * ALPHA_LIENZO +
            frame[idx_mask]  * (1.0 - ALPHA_LIENZO),
            0, 255
        ).astype(np.uint8)

        # ── Cursor del índice ──
        if cx_int is not None:
            dibujar_cursor(frame_final, cx_int, cy_int,
                           gesto_confirmado, PALETA_COLORES[idx_color])

        # ── UI ──
        dibujar_leyenda_gestos(frame_final, gesto_confirmado, estabiliz.progreso)
        dibujar_hud(frame_final, fps_mostrar, gesto_confirmado,
                    NOMBRES_COLORES[idx_color], grosor,
                    pausado if detectado else False)

        # Overlay de pausa
        if detectado and pausado:
            overlay = frame_final.copy()
            cv2.rectangle(overlay, (0, 0),
                          (ancho_frame, alto_frame), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame_final, 0.65, 0, frame_final)
            cv2.putText(frame_final, "PAUSADO  ✊",
                        (ancho_frame // 2 - 120, alto_frame // 2),
                        FUENTE, 1.4, (80, 80, 80), 3, cv2.LINE_AA)

        # Feedback de disparo
        if feedback_frames > 0:
            dibujar_feedback_disparo(frame_final, feedback_msg, feedback_color)
            feedback_frames -= 1

        # ── FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tf  = cv2.getTickCount()
            seg = (tf - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        cv2.imshow(NOMBRE_VENTANA, frame_final)

        # ── Teclado (respaldo) ──
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla == ord('s'):
            guardar_captura(frame_final)
        elif tecla == ord('r'):
            lienzo[:] = 0
            cx_prev = cy_prev = None
            print("  Lienzo limpiado.")
        elif tecla == ord('c'):
            idx_color = (idx_color + 1) % len(PALETA_COLORES)
            print(f"  Color: {NOMBRES_COLORES[idx_color]}")
        elif tecla in (ord('+'), ord('=')):
            grosor = min(grosor + 2, GROSOR_MAX)
            print(f"  Grosor: {grosor}px")
        elif tecla == ord('-'):
            grosor = max(grosor - 2, GROSOR_MIN)
            print(f"  Grosor: {grosor}px")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
