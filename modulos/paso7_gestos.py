"""
========================================================
  PASO 7 - Detectar Gestos
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Reconocer gestos simples de la mano usando distancias
    y ángulos entre landmarks, sin modelos extra.
    Los gestos controlan la pizarra digital del paso 6
    sin necesidad de teclado.

  Gestos implementados:
    ✊  PUÑO       → todos los dedos cerrados
    ✋  PALMA      → todos los dedos extendidos
    ☝️  ÍNDICE     → solo el índice extendido (puntero)
    🤏  PINCH      → pulgar + índice juntos (< umbral)
    ✌️  VICTORIA   → índice + medio extendidos
    🤟  OK         → pulgar + índice forman un círculo

  Acciones vinculadas al pinch:
    · Pinch activa/desactiva el modo dibujo en la pizarra

  Controles de teclado (respaldo):
    [ESPACIO] / [d]  Toggle dibujo
    [b]              Toggle borrador
    [c]              Siguiente color
    [+] [-]          Grosor pincel
    [r]              Limpiar lienzo
    [s]              Captura
    [q]              Salir

  Librerías requeridas:
    pip install opencv-python mediapipe
========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import os
from collections import deque
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
NOMBRE_VENTANA = "TouchWall - Paso 7: Gestos"

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

# Suavizado
LERP_FACTOR = 0.4

# Umbrales de gestos (en coordenadas normalizadas 0-1)
UMBRAL_PINCH        = 0.055   # dist pulgar-índice para considerar pinch
UMBRAL_DEDO_ABIERTO = 0.40    # relación punta/nudillo para dedo extendido

# Estabilizador: el gesto debe mantenerse N frames antes de activarse
FRAMES_CONFIRMACION = 6

# Pizarra
PALETA_COLORES = [
    (255, 255, 255),    # blanco
    (0,   0,   255),    # rojo
    (0,   165, 255),    # naranja
    (0,   255, 255),    # amarillo
    (0,   255,   0),    # verde
    (255, 0,     0),    # azul
    (255, 0,   255),    # magenta
    (0,   255, 200),    # cian
]
NOMBRES_COLORES = [
    "Blanco","Rojo","Naranja","Amarillo","Verde","Azul","Magenta","Cian"
]
GROSOR_INICIAL  = 8
GROSOR_MIN      = 2
GROSOR_MAX      = 50
GROSOR_PASO     = 2
GROSOR_BORRADOR = 40
ALPHA_LIENZO    = 0.75

FUENTE         = cv2.FONT_HERSHEY_SIMPLEX
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# ÍNDICES DE LANDMARKS RELEVANTES
# ─────────────────────────────────────────────────────
#
#  Muñeca     : 0
#  Pulgar     : 1(CMC) 2(MCP) 3(IP) 4(TIP)
#  Índice     : 5(MCP) 6(PIP) 7(DIP) 8(TIP)
#  Medio      : 9(MCP) 10(PIP) 11(DIP) 12(TIP)
#  Anular     : 13(MCP) 14(PIP) 15(DIP) 16(TIP)
#  Meñique    : 17(MCP) 18(PIP) 19(DIP) 20(TIP)

LM_MUNECA   = 0
LM_PULGAR   = [1, 2, 3, 4]
LM_INDICE   = [5, 6, 7, 8]
LM_MEDIO    = [9, 10, 11, 12]
LM_ANULAR   = [13, 14, 15, 16]
LM_MENIQUE  = [17, 18, 19, 20]

# Tuplas (punta, base_MCP) para cada dedo
DEDOS = [
    (4,  2,  "pulgar"),   # pulgar: punta vs MCP
    (8,  5,  "indice"),
    (12, 9,  "medio"),
    (16, 13, "anular"),
    (20, 17, "menique"),
]


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
        print(f"  [ERROR] No se pudo descargar: {e}")
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

    ancho_r = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_r  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_r   = cap.get(cv2.CAP_PROP_FPS)
    print("=" * 54)
    print("  TouchWall — Paso 7: Detectar Gestos")
    print("=" * 54)
    print(f"  Cámara  : {ancho_r}x{alto_r} @ {fps_r:.0f}fps")
    print(f"  Pinch umbral : {UMBRAL_PINCH}")
    print("=" * 54)
    print("  Gestos: PINCH=dibujo | PALMA=borrar trazo activo")
    print("  [d/ESPACIO] Toggle | [b] Borrador | [c] Color")
    print("  [+/-] Grosor | [r] Reset | [s] Captura | [q] Salir")
    print("=" * 54)
    return cap


def dist(lm_a, lm_b):
    """Distancia euclidiana entre dos landmarks normalizados."""
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)


def lerp(actual, objetivo, factor):
    return actual + (objetivo - actual) * factor


# ─────────────────────────────────────────────────────
# DETECCIÓN DE GESTOS
# ─────────────────────────────────────────────────────

def dedo_extendido(lm, punta_idx, mcp_idx):
    """
    Retorna True si el dedo está extendido.
    Compara si la punta está más lejos de la muñeca que el nudillo MCP.
    """
    punta = lm[punta_idx]
    mcp   = lm[mcp_idx]
    muneca = lm[LM_MUNECA]

    dist_punta  = dist(punta,  muneca)
    dist_mcp    = dist(mcp,    muneca)

    # Punta debe estar más lejos que el MCP × factor
    return dist_punta > dist_mcp * UMBRAL_DEDO_ABIERTO


def pulgar_extendido(lm):
    """
    El pulgar es especial: comparamos si su punta (LM4) está
    suficientemente lejos del nudillo CMC (LM1).
    """
    punta  = lm[4]
    cmc    = lm[1]
    return dist(punta, cmc) > 0.12


def detectar_gesto(lm):
    """
    Analiza los 21 landmarks y retorna el nombre del gesto detectado.

    Retorna uno de:
        'PINCH'    - pulgar + índice juntos
        'PUÑO'     - todos los dedos cerrados
        'PALMA'    - todos los dedos extendidos
        'INDICE'   - solo índice extendido (puntero)
        'VICTORIA' - índice + medio extendidos
        'OK'       - pulgar + índice forman círculo
        'OTRO'     - no reconocido
    """
    # Estado de cada dedo (True = extendido)
    ext_pulgar  = pulgar_extendido(lm)
    ext_indice  = dedo_extendido(lm, 8,  5)
    ext_medio   = dedo_extendido(lm, 12, 9)
    ext_anular  = dedo_extendido(lm, 16, 13)
    ext_menique = dedo_extendido(lm, 20, 17)

    # Distancia pulgar-índice para pinch
    d_pinch = dist(lm[4], lm[8])

    # ── Pinch: pulgar e índice muy juntos ──
    if d_pinch < UMBRAL_PINCH:
        return "PINCH"

    # ── Palma abierta: todos extendidos ──
    if ext_indice and ext_medio and ext_anular and ext_menique:
        return "PALMA"

    # ── Puño: todos cerrados ──
    if (not ext_indice and not ext_medio
            and not ext_anular and not ext_menique):
        return "PUÑO"

    # ── Victoria: índice + medio extendidos, resto cerrados ──
    if (ext_indice and ext_medio
            and not ext_anular and not ext_menique):
        return "VICTORIA"

    # ── Solo índice extendido (puntero) ──
    if (ext_indice and not ext_medio
            and not ext_anular and not ext_menique):
        return "INDICE"

    # ── OK: pulgar + índice círculo (no pinch, pero próximos) ──
    if ext_pulgar and not ext_indice and ext_medio:
        return "OK"

    return "OTRO"


# ─────────────────────────────────────────────────────
# DIBUJO
# ─────────────────────────────────────────────────────

# Emojis/textos e info de cada gesto
INFO_GESTOS = {
    "PINCH":    ("🤏 PINCH",    "Toggle dibujo",  (0, 220, 255)),
    "PALMA":    ("✋ PALMA",    "Borra ultimo",   (100, 220, 100)),
    "PUÑO":     ("✊ PUÑO",     "Nada",           (80,  80,  80)),
    "VICTORIA": ("✌️ VICTORIA", "Nada",           (200, 100, 255)),
    "INDICE":   ("☝️  INDICE",  "Puntero",        (0,  200, 255)),
    "OK":       ("OK",          "Nada",           (255, 180,  0)),
    "OTRO":     ("… OTRO",      "—",              (120, 120, 120)),
}


def dibujar_esqueleto(frame, lm, alto, ancho, modo_dibujo):
    """Dibuja el esqueleto de la mano; verde si dibujando, gris si no."""
    color_con  = (0, 200, 80)  if modo_dibujo else (60, 60, 60)
    color_lm   = (0, 240, 100) if modo_dibujo else (80, 80, 80)

    puntos = [(int(l.x * ancho), int(l.y * alto)) for l in lm]

    for con in CONEXIONES_MANO:
        s, e = con.start, con.end
        if s < len(puntos) and e < len(puntos):
            cv2.line(frame, puntos[s], puntos[e], color_con, 1)

    for i, (cx, cy) in enumerate(puntos):
        r = 6 if i in [4, 8, 12, 16, 20] else 3
        cv2.circle(frame, (cx, cy), r, color_lm, -1)

    return puntos


def dibujar_marcador_indice(frame, cx, cy, modo_dibujo, color_pincel):
    """Cursor sobre la punta del índice."""
    color = color_pincel if modo_dibujo else (180, 180, 180)
    cv2.circle(frame, (cx, cy), 16, color, 2)
    cv2.circle(frame, (cx, cy),  7, color, -1)


def dibujar_indicador_gesto(frame, gesto, confirmado, progreso):
    """
    Panel lateral derecho con el gesto actual y barra de progreso
    de confirmación (evita activaciones accidentales).
    """
    alto_f, ancho_f = frame.shape[:2]
    label, accion, color = INFO_GESTOS.get(gesto, INFO_GESTOS["OTRO"])

    # Fondo panel
    px = ancho_f - 240
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 0), (ancho_f, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, label,  (px + 10, 32),  FUENTE, 0.75, color, 2)
    cv2.putText(frame, accion, (px + 10, 56),  FUENTE, 0.52, color, 1)

    # Barra de progreso de confirmación
    bw = 200
    bh = 12
    bx = px + 10
    by = 72
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (40, 40, 40), -1)
    fill = int(bw * progreso)
    if fill > 0:
        c_barra = (0, 200, 50) if confirmado else color
        cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), c_barra, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (80, 80, 80), 1)
    cv2.putText(frame, "confirmando..." if not confirmado else "ACTIVO",
                (bx, by + bh + 16), FUENTE, 0.42,
                (0, 200, 50) if confirmado else (130, 130, 130), 1)


def dibujar_paleta(frame, idx_color, grosor):
    alto_f, ancho_f = frame.shape[:2]
    tam    = 30
    margen = 8
    y_base = alto_f - tam - margen

    for i, color in enumerate(PALETA_COLORES):
        x = margen + i * (tam + 4)
        cv2.rectangle(frame, (x, y_base), (x + tam, y_base + tam), (20, 20, 20), -1)
        cv2.rectangle(frame, (x + 3, y_base + 3),
                      (x + tam - 3, y_base + tam - 3), color, -1)
        if i == idx_color:
            cv2.rectangle(frame, (x, y_base), (x + tam, y_base + tam),
                          (255, 255, 255), 2)

    gx = margen + len(PALETA_COLORES) * (tam + 4) + 10
    gy = y_base + tam // 2
    cv2.line(frame, (gx, gy), (gx + 40, gy), PALETA_COLORES[idx_color], max(1, grosor // 3))
    cv2.putText(frame, f"{grosor}px", (gx + 46, gy + 5),
                FUENTE, 0.45, (180, 180, 180), 1)


def dibujar_hud(frame, fps, detectado, modo_dibujo, borrando, nombre_color):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    color = (0, 220, 100) if modo_dibujo else (100, 100, 220)
    estado = "DIBUJANDO" if modo_dibujo else ("Borrando" if borrando else "Puntero")

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 26), FUENTE, 0.8, color, 2)
    cv2.putText(frame, f"Modo: {estado}  |  Color: {nombre_color}",
                (10, 52), FUENTE, 0.6, color, 1)
    cv2.putText(frame, "[d/SPC] toggle  [b] borrador  [c] color  [r] reset  [q] salir",
                (10, 72), FUENTE, 0.43, (140, 140, 140), 1)

    alto_f, ancho_f = frame.shape[:2]
    cv2.circle(frame, (ancho_f - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_f - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)


def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso7_{ts}.png")
    cv2.imwrite(ruta, frame)
    print(f"  [OK] Captura guardada: {ruta}")


# ─────────────────────────────────────────────────────
# ESTABILIZADOR DE GESTOS
# ─────────────────────────────────────────────────────

class EstabilizadorGesto:
    """
    Evita activaciones accidentales: el gesto debe mantenerse
    FRAMES_CONFIRMACION frames consecutivos para confirmar.
    """
    def __init__(self, frames_necesarios=FRAMES_CONFIRMACION):
        self.frames_necesarios = frames_necesarios
        self.gesto_actual      = "OTRO"
        self.contador          = 0
        self.gesto_confirmado  = None
        self.nuevo_evento      = False   # True solo el frame de activación

    def actualizar(self, gesto_raw):
        self.nuevo_evento = False

        if gesto_raw == self.gesto_actual:
            self.contador = min(self.contador + 1, self.frames_necesarios)
        else:
            self.gesto_actual = gesto_raw
            self.contador     = 1

        # ¿Se acaba de confirmar?
        if (self.contador == self.frames_necesarios and
                self.gesto_actual != self.gesto_confirmado):
            self.gesto_confirmado = self.gesto_actual
            self.nuevo_evento     = True

        return self.gesto_confirmado

    @property
    def progreso(self):
        return self.contador / self.frames_necesarios


# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    if not verificar_modelo():
        return

    cap = iniciar_camara()
    if cap is None:
        return

    detector    = crear_detector()
    estabiliz   = EstabilizadorGesto()

    # Estado de pizarra
    lienzo         = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    idx_color      = 0
    grosor         = GROSOR_INICIAL
    modo_dibujo    = False
    borrando       = False
    cx_prev = cy_prev = None

    cx_suave = float(ANCHO // 2)
    cy_suave = float(ALTO  // 2)

    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()
    timestamp_ms    = 0

    print("  Listo. Muestra la mano y haz PINCH para dibujar.")

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

        detectado   = False
        gesto_raw   = "OTRO"
        cx_int = cy_int = None

        if resultado.hand_landmarks:
            lm_list  = resultado.hand_landmarks[0]
            detectado = True

            # Suavizado del índice
            cx_suave = lerp(cx_suave, lm_list[INDICE_LM].x * ancho_frame, LERP_FACTOR)
            cy_suave = lerp(cy_suave, lm_list[INDICE_LM].y * alto_frame,  LERP_FACTOR)
            cx_int   = int(cx_suave)
            cy_int   = int(cy_suave)

            # Detectar gesto
            gesto_raw = detectar_gesto(lm_list)

            # Estabilizar
            gesto_confirmado = estabiliz.actualizar(gesto_raw)

            # ── Reaccionar a gestos nuevos ──
            if estabiliz.nuevo_evento:
                if gesto_confirmado == "PINCH":
                    modo_dibujo = not modo_dibujo
                    borrando    = False
                    cx_prev = cy_prev = None
                    estado = "ACTIVO ✓" if modo_dibujo else "PAUSADO"
                    print(f"  [GESTO] PINCH → modo dibujo {estado}")

            # ── Trazar en el lienzo ──
            if modo_dibujo and not borrando and cx_int is not None:
                color_actual = PALETA_COLORES[idx_color]
                if cx_prev is not None:
                    cv2.line(lienzo, (cx_prev, cy_prev),
                             (cx_int, cy_int), color_actual, grosor)
                else:
                    cv2.circle(lienzo, (cx_int, cy_int),
                               grosor // 2, color_actual, -1)
                cx_prev, cy_prev = cx_int, cy_int

            elif borrando and cx_int is not None:
                cv2.circle(lienzo, (cx_int, cy_int), GROSOR_BORRADOR, (0, 0, 0), -1)
                cx_prev, cy_prev = cx_int, cy_int

            else:
                cx_prev = cy_prev = None

            # Esqueleto y marcador
            dibujar_esqueleto(frame, lm_list, alto_frame, ancho_frame, modo_dibujo)
            if cx_int is not None:
                dibujar_marcador_indice(frame, cx_int, cy_int,
                                        modo_dibujo, PALETA_COLORES[idx_color])

            # Panel de gesto
            dibujar_indicador_gesto(frame, gesto_raw,
                                    estabiliz.contador >= estabiliz.frames_necesarios,
                                    estabiliz.progreso)
        else:
            cx_prev = cy_prev = None
            estabiliz.actualizar("OTRO")

        # ── Componer lienzo sobre video ──
        mascara = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(mascara, 1, 255, cv2.THRESH_BINARY)
        frame_final = frame.copy()
        idx = mascara > 0
        frame_final[idx] = np.clip(
            lienzo[idx] * ALPHA_LIENZO + frame[idx] * (1.0 - ALPHA_LIENZO),
            0, 255
        ).astype(np.uint8)

        # ── UI ──
        dibujar_paleta(frame_final, idx_color, grosor)
        dibujar_hud(frame_final, fps_mostrar, detectado,
                    modo_dibujo, borrando, NOMBRES_COLORES[idx_color])

        # ── FPS ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin    = cv2.getTickCount()
            tiempo_seg    = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar   = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        cv2.imshow(NOMBRE_VENTANA, frame_final)

        # ── Controles de teclado (respaldo) ──
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla in (ord(' '), ord('d')):
            modo_dibujo = not modo_dibujo
            borrando    = False
            cx_prev = cy_prev = None
            print(f"  Dibujo: {'ACTIVO' if modo_dibujo else 'PAUSADO'}")
        elif tecla == ord('b'):
            borrando = not borrando
            if borrando:
                modo_dibujo = False
            cx_prev = cy_prev = None
            print(f"  Borrador: {'ON' if borrando else 'OFF'}")
        elif tecla == ord('c'):
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
