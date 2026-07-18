"""
========================================================
  PASO 8 - Botones Virtuales
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Crear botones virtuales en pantalla que se activan
    cuando la punta del dedo índice entra en su área
    y se confirman con un PINCH (gesto del paso 7).

  Sistema de botones:
    · El dedo entra en el área del botón  → hover (resaltado)
    · Se mantiene el hover N frames       → barra de carga
    · Barra llena                         → acción ejecutada
    (Sin necesidad de pinch para mayor ergonomía)

  Botones disponibles (barra lateral izquierda):
    [COLOR]    × 8  Cambiar color del pincel
    [BORRADOR]      Activar/desactivar borrador
    [LIMPIAR]       Limpiar todo el lienzo
    [GROSOR+]       Aumentar grosor del pincel
    [GROSOR-]       Reducir grosor del pincel
    [DIBUJAR]       Toggle modo dibujo (igual que PINCH)

  Gestos de la mano:
    PINCH → toggle modo dibujo

  Teclado (respaldo):
    [d/SPC] Dibujo | [b] Borrador | [r] Reset
    [c] Color  [+/-] Grosor  [s] Captura  [q] Salir

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
NOMBRE_VENTANA = "TouchWall - Paso 8: Botones Virtuales"

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
LERP_FACTOR = 0.35

# Gestos
UMBRAL_PINCH        = 0.055
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
GROSOR_PASO     = 3
GROSOR_BORRADOR = 40
ALPHA_LIENZO    = 0.75

# Botones — hover dwell time (frames para activar sin pinch)
FRAMES_HOVER    = 25      # ~0.83s a 30fps

# Dimensiones de la barra de botones
BTN_X       = 10          # margen izquierdo
BTN_W       = 130         # ancho de cada botón
BTN_H       = 48          # alto de cada botón
BTN_GAP     = 6           # separación entre botones
BTN_RADIO   = 8           # radio de esquinas redondeadas

FUENTE         = cv2.FONT_HERSHEY_SIMPLEX
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# CLASE BOTÓN VIRTUAL
# ─────────────────────────────────────────────────────

class BotonVirtual:
    """
    Botón rectangular con sistema de dwell (permanencia del dedo).
    Se activa cuando el índice está dentro del área BTN_HOVER_FRAMES
    frames consecutivos.
    """
    def __init__(self, x, y, w, h, etiqueta, accion_id,
                 color_normal=(40, 40, 40),
                 color_hover=(60, 80, 110),
                 color_activo=(0, 180, 80),
                 color_texto=(220, 220, 220)):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.etiqueta    = etiqueta
        self.accion_id   = accion_id
        self.color_normal = color_normal
        self.color_hover  = color_hover
        self.color_activo = color_activo
        self.color_texto  = color_texto

        self.hover_frames  = 0
        self.activo        = False      # se activa un frame, luego se limpia
        self.ultimo_activo = False      # para animación de feedback

    def contiene(self, cx, cy):
        return (self.x <= cx <= self.x + self.w and
                self.y <= cy <= self.y + self.h)

    def actualizar(self, cx, cy):
        """Llama cada frame con la posición del índice."""
        self.activo = False
        if self.contiene(cx, cy):
            self.hover_frames += 1
            if self.hover_frames >= FRAMES_HOVER:
                self.activo       = True
                self.ultimo_activo = True
                self.hover_frames = 0    # reiniciar para evitar activación continua
        else:
            self.hover_frames  = 0
            self.ultimo_activo = False

    @property
    def progreso_hover(self):
        return min(self.hover_frames / FRAMES_HOVER, 1.0)

    def dibujar(self, frame, color_override=None):
        x, y, w, h = self.x, self.y, self.w, self.h
        progreso = self.progreso_hover

        # Fondo del botón
        if self.ultimo_activo:
            bg_color = self.color_activo
        elif progreso > 0:
            # Interpolar entre normal y hover
            bg_color = tuple(
                int(self.color_normal[i] * (1 - progreso) +
                    self.color_hover[i]  * progreso)
                for i in range(3)
            )
        else:
            bg_color = color_override if color_override else self.color_normal

        # Rectángulo redondeado (simulado con círculos en esquinas)
        _rect_redondeado(frame, (x, y), (x + w, y + h), bg_color, BTN_RADIO, -1)

        # Borde
        borde_color = (100, 160, 220) if progreso > 0 else (70, 70, 70)
        _rect_redondeado(frame, (x, y), (x + w, y + h), borde_color, BTN_RADIO, 1)

        # Barra de progreso dwell (parte inferior del botón)
        if 0 < progreso < 1.0:
            bx1 = x + 2
            bx2 = x + 2 + int((w - 4) * progreso)
            by  = y + h - 5
            cv2.line(frame, (bx1, by), (bx2, by), (0, 200, 255), 3)

        # Texto
        texto = self.etiqueta
        ts    = cv2.getTextSize(texto, FUENTE, 0.50, 1)[0]
        tx    = x + (w - ts[0]) // 2
        ty    = y + (h + ts[1]) // 2
        cv2.putText(frame, texto, (tx, ty), FUENTE, 0.50,
                    self.color_texto, 1, cv2.LINE_AA)


def _rect_redondeado(frame, pt1, pt2, color, radio, grosor):
    """Dibuja un rectángulo con esquinas redondeadas."""
    x1, y1 = pt1
    x2, y2 = pt2
    r = radio

    if grosor == -1:  # relleno
        cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(frame, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(frame, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(frame, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(frame, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, grosor)
        cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, grosor)
        cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, grosor)
        cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, grosor)
        cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, grosor)
        cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, grosor)
        cv2.ellipse(frame, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, grosor)
        cv2.ellipse(frame, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, grosor)


# ─────────────────────────────────────────────────────
# CONSTRUCCIÓN DE LA BARRA DE BOTONES
# ─────────────────────────────────────────────────────

def construir_botones():
    """Crea la lista de BotonVirtual para la barra lateral izquierda."""
    botones = []
    y = 10

    def agregar(etiqueta, accion_id, color_n=(35,35,35),
                color_h=(55,80,110), color_a=(0,180,80)):
        nonlocal y
        b = BotonVirtual(BTN_X, y, BTN_W, BTN_H,
                         etiqueta, accion_id,
                         color_normal=color_n,
                         color_hover=color_h,
                         color_activo=color_a)
        botones.append(b)
        y += BTN_H + BTN_GAP
        return b

    # Modo dibujo
    agregar("✏  DIBUJAR", "toggle_dibujo",
            color_h=(40, 100, 50), color_a=(0, 200, 80))
    agregar("⬜  BORRADOR", "toggle_borrador",
            color_h=(70, 70, 70),  color_a=(160, 160, 160))
    agregar("🗑  LIMPIAR",  "limpiar",
            color_h=(120, 40, 40), color_a=(200, 60, 60))

    y += 8  # separador

    # Grosor
    agregar("▲  GROSOR+", "grosor_mas",
            color_h=(50, 80, 50),  color_a=(80, 180, 80))
    agregar("▼  GROSOR-", "grosor_menos",
            color_h=(50, 80, 50),  color_a=(80, 180, 80))

    y += 8  # separador

    # Colores
    for i, color in enumerate(PALETA_COLORES):
        b = BotonVirtual(
            BTN_X, y, BTN_W, BTN_H,
            f"  {NOMBRES_COLORES[i]}", f"color_{i}",
            color_normal=tuple(max(0, c // 4) for c in color),
            color_hover =tuple(c // 2 for c in color),
            color_activo=color,
            color_texto =(255, 255, 255),
        )
        botones.append(b)
        y += BTN_H + BTN_GAP

    return botones


# ─────────────────────────────────────────────────────
# DETECCIÓN DE GESTOS (del paso 7)
# ─────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def detectar_pinch(lm):
    return dist(lm[4], lm[8]) < UMBRAL_PINCH


class EstabilizadorPinch:
    def __init__(self, n=FRAMES_CONFIRMACION):
        self.n          = n
        self.contador   = 0
        self.confirmado = False
        self.nuevo      = False

    def actualizar(self, es_pinch):
        self.nuevo = False
        if es_pinch:
            self.contador += 1
        else:
            self.contador = 0
            self.confirmado = False

        if self.contador >= self.n and not self.confirmado:
            self.confirmado = True
            self.nuevo      = True

        return self.confirmado


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

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("=" * 56)
    print("  TouchWall — Paso 8: Botones Virtuales")
    print("=" * 56)
    print(f"  Cámara  : {aw}x{ah} @ {fps:.0f}fps")
    print(f"  Dwell   : {FRAMES_HOVER} frames (~{FRAMES_HOVER/fps:.1f}s) para activar")
    print("=" * 56)
    print("  Apunta el dedo al botón y espera la barra de carga")
    print("  PINCH = toggle dibujo")
    print("=" * 56)
    return cap


def lerp(actual, objetivo, factor):
    return actual + (objetivo - actual) * factor


def guardar_captura(frame):
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta = os.path.join(carpeta, f"captura_paso8_{ts}.png")
    cv2.imwrite(ruta, frame)
    print(f"  [OK] Captura guardada: {ruta}")


# ─────────────────────────────────────────────────────
# DIBUJO UI
# ─────────────────────────────────────────────────────

def dibujar_esqueleto_tenue(frame, lm, alto, ancho):
    puntos = [(int(l.x * ancho), int(l.y * alto)) for l in lm]
    for con in CONEXIONES_MANO:
        s, e = con.start, con.end
        if s < len(puntos) and e < len(puntos):
            cv2.line(frame, puntos[s], puntos[e], (55, 55, 55), 1)
    for i, (cx, cy) in enumerate(puntos):
        if i != INDICE_LM:
            cv2.circle(frame, (cx, cy), 3, (75, 75, 75), -1)


def dibujar_cursor_indice(frame, cx, cy, modo_dibujo, color_pincel):
    color = color_pincel if modo_dibujo else (200, 200, 200)
    cv2.circle(frame, (cx, cy), 16, color, 2)
    cv2.circle(frame, (cx, cy),  7, color, -1)
    cv2.line(frame, (cx - 22, cy), (cx + 22, cy), color, 1)
    cv2.line(frame, (cx, cy - 22), (cx, cy + 22), color, 1)


def dibujar_hud(frame, fps, modo_dibujo, borrando, grosor, nombre_color):
    alto_f, ancho_f = frame.shape[:2]
    color = (0, 220, 100) if modo_dibujo else (100, 100, 220)

    # Panel inferior derecho
    panel_x = BTN_X + BTN_W + 10
    panel_w = ancho_f - panel_x - 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, alto_f - 50),
                  (ancho_f - 10, alto_f - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    estado = "DIBUJANDO" if modo_dibujo else ("BORRADOR" if borrando else "PUNTERO")
    cv2.putText(frame,
                f"FPS:{fps:.0f}  |  {estado}  |  Color:{nombre_color}  |  Grosor:{grosor}px",
                (panel_x + 8, alto_f - 20), FUENTE, 0.52, color, 1)

    cv2.putText(frame,
                "[d/SPC]toggle [b]borrador [c]color [+/-]grosor [r]reset [s]captura [q]salir",
                (panel_x + 8, alto_f - 8), FUENTE, 0.38, (120, 120, 120), 1)

    # LIVE
    cv2.circle(frame, (ancho_f - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ancho_f - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)


def dibujar_fondo_barra(frame, n_botones):
    """Fondo semitransparente detrás de la barra de botones."""
    alto_barra = n_botones * (BTN_H + BTN_GAP) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (BTN_X + BTN_W + 8, alto_barra),
                  (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


# ─────────────────────────────────────────────────────
# PROCESAR ACCIÓN DE BOTÓN
# ─────────────────────────────────────────────────────

def procesar_accion(accion_id, estado, lienzo):
    """
    Ejecuta la acción del botón activado.
    `estado` es un dict mutable con las variables de la pizarra.
    """
    if accion_id == "toggle_dibujo":
        estado["modo_dibujo"] = not estado["modo_dibujo"]
        estado["borrando"]    = False
        estado["cx_prev"] = estado["cy_prev"] = None
        print(f"  [BTN] Dibujo: {'ACTIVO' if estado['modo_dibujo'] else 'PAUSADO'}")

    elif accion_id == "toggle_borrador":
        estado["borrando"]    = not estado["borrando"]
        estado["modo_dibujo"] = False
        estado["cx_prev"] = estado["cy_prev"] = None
        print(f"  [BTN] Borrador: {'ON' if estado['borrando'] else 'OFF'}")

    elif accion_id == "limpiar":
        lienzo[:] = 0
        estado["cx_prev"] = estado["cy_prev"] = None
        print("  [BTN] Lienzo limpiado.")

    elif accion_id == "grosor_mas":
        estado["grosor"] = min(estado["grosor"] + GROSOR_PASO, GROSOR_MAX)
        print(f"  [BTN] Grosor: {estado['grosor']}px")

    elif accion_id == "grosor_menos":
        estado["grosor"] = max(estado["grosor"] - GROSOR_PASO, GROSOR_MIN)
        print(f"  [BTN] Grosor: {estado['grosor']}px")

    elif accion_id.startswith("color_"):
        idx = int(accion_id.split("_")[1])
        estado["idx_color"] = idx
        print(f"  [BTN] Color: {NOMBRES_COLORES[idx]}")


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
    estabiliz = EstabilizadorPinch()
    botones   = construir_botones()

    # Estado de pizarra (dict mutable para pasar por referencia)
    lienzo = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    estado = {
        "idx_color":  0,
        "grosor":     GROSOR_INICIAL,
        "modo_dibujo": False,
        "borrando":   False,
        "cx_prev":    None,
        "cy_prev":    None,
    }

    cx_suave = float(ANCHO // 2)
    cy_suave = float(ALTO  // 2)

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

        cx_int = cy_int = None

        if resultado.hand_landmarks:
            lm = resultado.hand_landmarks[0]
            dibujar_esqueleto_tenue(frame, lm, alto_frame, ancho_frame)

            # Posición suavizada del índice
            cx_suave = lerp(cx_suave, lm[INDICE_LM].x * ancho_frame, LERP_FACTOR)
            cy_suave = lerp(cy_suave, lm[INDICE_LM].y * alto_frame,  LERP_FACTOR)
            cx_int   = int(cx_suave)
            cy_int   = int(cy_suave)

            # Pinch → toggle dibujo
            estabiliz.actualizar(detectar_pinch(lm))
            if estabiliz.nuevo:
                procesar_accion("toggle_dibujo", estado, lienzo)

            # Actualizar botones y procesar activaciones
            for btn in botones:
                btn.actualizar(cx_int, cy_int)
                if btn.activo:
                    procesar_accion(btn.accion_id, estado, lienzo)

            # Trazar en el lienzo
            if estado["modo_dibujo"] and not estado["borrando"]:
                color_actual = PALETA_COLORES[estado["idx_color"]]
                if estado["cx_prev"] is not None:
                    cv2.line(lienzo,
                             (estado["cx_prev"], estado["cy_prev"]),
                             (cx_int, cy_int),
                             color_actual, estado["grosor"])
                else:
                    cv2.circle(lienzo, (cx_int, cy_int),
                               estado["grosor"] // 2, color_actual, -1)
                estado["cx_prev"], estado["cy_prev"] = cx_int, cy_int

            elif estado["borrando"]:
                cv2.circle(lienzo, (cx_int, cy_int), GROSOR_BORRADOR, (0, 0, 0), -1)
                estado["cx_prev"], estado["cy_prev"] = cx_int, cy_int

            else:
                estado["cx_prev"] = estado["cy_prev"] = None

        else:
            estado["cx_prev"] = estado["cy_prev"] = None
            for btn in botones:
                btn.actualizar(-1, -1)    # fuera de todos los botones

        # ── Componer lienzo + video ──
        mascara = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
        _, mascara = cv2.threshold(mascara, 1, 255, cv2.THRESH_BINARY)
        frame_final = frame.copy()
        idx = mascara > 0
        frame_final[idx] = np.clip(
            lienzo[idx] * ALPHA_LIENZO + frame[idx] * (1.0 - ALPHA_LIENZO),
            0, 255
        ).astype(np.uint8)

        # ── Barra de botones ──
        dibujar_fondo_barra(frame_final, len(botones))
        color_pincel = PALETA_COLORES[estado["idx_color"]]
        for btn in botones:
            # Los botones de color muestran su propio color como fondo cuando están inactivos
            if btn.accion_id.startswith("color_"):
                idx_c = int(btn.accion_id.split("_")[1])
                c = PALETA_COLORES[idx_c]
                override = tuple(max(20, v // 3) for v in c)
                btn.dibujar(frame_final, override)
            else:
                btn.dibujar(frame_final)

        # ── Cursor del índice ──
        if cx_int is not None:
            dibujar_cursor_indice(frame_final, cx_int, cy_int,
                                  estado["modo_dibujo"], color_pincel)

        # ── HUD ──
        dibujar_hud(frame_final, fps_mostrar,
                    estado["modo_dibujo"], estado["borrando"],
                    estado["grosor"], NOMBRES_COLORES[estado["idx_color"]])

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
        elif tecla in (ord(' '), ord('d')):
            procesar_accion("toggle_dibujo", estado, lienzo)
        elif tecla == ord('b'):
            procesar_accion("toggle_borrador", estado, lienzo)
        elif tecla == ord('r'):
            procesar_accion("limpiar", estado, lienzo)
        elif tecla == ord('c'):
            estado["idx_color"] = (estado["idx_color"] + 1) % len(PALETA_COLORES)
            print(f"  Color: {NOMBRES_COLORES[estado['idx_color']]}")
        elif tecla in (ord('+'), ord('=')):
            procesar_accion("grosor_mas", estado, lienzo)
        elif tecla == ord('-'):
            procesar_accion("grosor_menos", estado, lienzo)
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
