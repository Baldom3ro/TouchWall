"""
========================================================
  PASO 8 - Botones y Gestos Integrados
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Integrar los gestos intuitivos con los botones virtuales.
    Los botones se activan inmediatamente haciendo PINCH
    sobre ellos, o esperando (Dwell) como respaldo.

  Nuevos Gestos (Solicitados por el usuario):
    👆 INDICE    → Mover cursor (hover de botones) sin dibujar
    🤏 PINCH     → Clic en botón / Dibujar en lienzo
    ✌️ DOS JUNTOS→ Borrador (índice y medio pegados)
    ✌️ VICTORIA  → Cambiar color (índice y medio separados)
    ✊ PUÑO      → Pausa total

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
NOMBRE_VENTANA = "TouchWall - Paso 8: Botones y Gestos"

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

# Umbrales
UMBRAL_PINCH   = 0.055
FRAMES_ESTABLE = 4       # frames para estabilizar el gesto detectado
FRAMES_HOVER   = 25      # dwell time como respaldo por si no funciona el pinch

# Pizarra
PALETA_COLORES = [
    (255, 255, 255), (0, 0, 255), (0, 165, 255), (0, 255, 255),
    (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 200)
]
NOMBRES_COLORES = [
    "Blanco","Rojo","Naranja","Amarillo","Verde","Azul","Magenta","Cian"
]
GROSOR_INICIAL  = 10
GROSOR_MIN      = 2
GROSOR_MAX      = 50
GROSOR_PASO     = 3
GROSOR_BORRADOR = 45
ALPHA_LIENZO    = 0.80

# Gestos
GESTO_INDICE   = "INDICE"
GESTO_PINCH    = "PINCH"
GESTO_VICTORIA = "VICTORIA"
GESTO_DOS_JUNTOS = "DOS JUNTOS"
GESTO_PUÑO     = "PUÑO"
GESTO_OTRO     = "OTRO"

# UI de botones
BTN_X = 10
BTN_W = 140
BTN_H = 48
BTN_GAP = 6
BTN_RADIO = 8
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
CONEXIONES_MANO = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS


# ─────────────────────────────────────────────────────
# CLASE BOTÓN VIRTUAL
# ─────────────────────────────────────────────────────

class BotonVirtual:
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
        self.activo        = False
        self.ultimo_activo = False

    def contiene(self, cx, cy):
        return (self.x <= cx <= self.x + self.w and
                self.y <= cy <= self.y + self.h)

    def actualizar(self, cx, cy, pinch_nuevo):
        """Si se hace pinch nuevo sobre el botón, hace clic inmediato."""
        self.activo = False
        hover = self.contiene(cx, cy)
        
        if hover:
            # Activación inmediata por PINCH
            if pinch_nuevo:
                self.activo       = True
                self.ultimo_activo = True
                self.hover_frames = 0
            else:
                # Activación por tiempo (respaldo)
                self.hover_frames += 1
                if self.hover_frames >= FRAMES_HOVER:
                    self.activo       = True
                    self.ultimo_activo = True
                    self.hover_frames = 0
        else:
            self.hover_frames  = 0
            self.ultimo_activo = False

    @property
    def progreso_hover(self):
        return min(self.hover_frames / FRAMES_HOVER, 1.0)

    def dibujar(self, frame, color_override=None):
        x, y, w, h = self.x, self.y, self.w, self.h
        progreso = self.progreso_hover

        if self.ultimo_activo:
            bg_color = self.color_activo
        elif progreso > 0:
            bg_color = tuple(
                int(self.color_normal[i] * (1 - progreso) +
                    self.color_hover[i]  * progreso)
                for i in range(3)
            )
        else:
            bg_color = color_override if color_override else self.color_normal

        _rect_redondeado(frame, (x, y), (x + w, y + h), bg_color, BTN_RADIO, -1)

        borde_color = (100, 160, 220) if progreso > 0 else (70, 70, 70)
        _rect_redondeado(frame, (x, y), (x + w, y + h), borde_color, BTN_RADIO, 1)

        if 0 < progreso < 1.0:
            bx1 = x + 2
            bx2 = x + 2 + int((w - 4) * progreso)
            by  = y + h - 5
            cv2.line(frame, (bx1, by), (bx2, by), (0, 200, 255), 3)

        ts = cv2.getTextSize(self.etiqueta, FUENTE, 0.50, 1)[0]
        tx = x + (w - ts[0]) // 2
        ty = y + (h + ts[1]) // 2
        cv2.putText(frame, self.etiqueta, (tx, ty), FUENTE, 0.50,
                    self.color_texto, 1, cv2.LINE_AA)

def _rect_redondeado(frame, pt1, pt2, color, radio, grosor):
    x1, y1 = pt1
    x2, y2 = pt2
    r = radio
    if grosor == -1:
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

def construir_botones():
    botones = []
    y = 10
    def agregar(etiq, acc_id, color_n=(35,35,35), color_h=(55,80,110), color_a=(0,180,80)):
        nonlocal y
        b = BotonVirtual(BTN_X, y, BTN_W, BTN_H, etiq, acc_id, color_n, color_h, color_a)
        botones.append(b)
        y += BTN_H + BTN_GAP
        return b
    
    agregar("🗑  LIMPIAR",  "limpiar", color_h=(120, 40, 40), color_a=(200, 60, 60))
    y += 8
    agregar("▲  GROSOR+", "grosor_mas", color_h=(50, 80, 50), color_a=(80, 180, 80))
    agregar("▼  GROSOR-", "grosor_menos", color_h=(50, 80, 50), color_a=(80, 180, 80))
    y += 8
    for i, color in enumerate(PALETA_COLORES):
        b = BotonVirtual(BTN_X, y, BTN_W, BTN_H, f"  {NOMBRES_COLORES[i]}", f"color_{i}",
                         color_normal=tuple(max(0, c//4) for c in color),
                         color_hover=tuple(c//2 for c in color),
                         color_activo=color, color_texto=(255,255,255))
        botones.append(b)
        y += BTN_H + BTN_GAP
    return botones

def dibujar_fondo_barra(frame, n_botones):
    alto_barra = n_botones * (BTN_H + BTN_GAP) + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (BTN_X + BTN_W + 8, alto_barra), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


# ─────────────────────────────────────────────────────
# DETECCIÓN DE GESTOS
# ─────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def _dedo_extendido(lm, punta, pip):
    """True si la punta está más lejos de la muñeca que la art. intermedia (PIP)."""
    return dist(lm[punta], lm[0]) > dist(lm[pip], lm[0])

def clasificar_gesto(lm) -> str:
    ext_pulgar  = dist(lm[4], lm[0]) > dist(lm[2], lm[0])
    ext_indice  = _dedo_extendido(lm, 8,  6)
    ext_medio   = _dedo_extendido(lm, 12, 10)
    ext_anular  = _dedo_extendido(lm, 16, 14)
    ext_menique = _dedo_extendido(lm, 20, 18)
    d_pinch        = dist(lm[4], lm[8])
    d_indice_medio = dist(lm[8], lm[12])

    if d_pinch < UMBRAL_PINCH:
        return GESTO_PINCH
    if not ext_indice and not ext_medio and not ext_anular and not ext_menique:
        return GESTO_PUÑO
    if ext_indice and ext_medio and not ext_anular and not ext_menique:
        if d_indice_medio < 0.055:  # dedos pegados
            return GESTO_DOS_JUNTOS
        else:                       # dedos separados (V)
            return GESTO_VICTORIA
    if ext_indice and not ext_medio and not ext_anular and not ext_menique:
        return GESTO_INDICE
    return GESTO_OTRO

class Estabilizador:
    GESTOS_DISPARO = {GESTO_VICTORIA}
    def __init__(self, n=FRAMES_ESTABLE):
        self.n = n
        self.gesto_bruto = GESTO_OTRO
        self.contador = 0
        self.confirmado = GESTO_OTRO
        self.disparo = False
        self.nuevo = False
    def actualizar(self, gesto_nuevo):
        self.disparo = False
        self.nuevo   = False
        if gesto_nuevo == self.gesto_bruto:
            self.contador = min(self.contador + 1, self.n)
        else:
            self.gesto_bruto = gesto_nuevo
            self.contador = 1
        
        nuevo_confirmado = self.gesto_bruto if self.contador >= self.n else self.confirmado
        
        if nuevo_confirmado != self.confirmado:
            self.confirmado = nuevo_confirmado
            self.nuevo = True
            if nuevo_confirmado in self.GESTOS_DISPARO:
                self.disparo = True
        return self.confirmado


# ─────────────────────────────────────────────────────
# UI Y UTILIDADES
# ─────────────────────────────────────────────────────

def procesar_accion(accion_id, estado, lienzo):
    if accion_id == "limpiar":
        lienzo[:] = 0
        print("  [BTN] Lienzo limpiado.")
    elif accion_id == "grosor_mas":
        estado["grosor"] = min(estado["grosor"] + GROSOR_PASO, GROSOR_MAX)
    elif accion_id == "grosor_menos":
        estado["grosor"] = max(estado["grosor"] - GROSOR_PASO, GROSOR_MIN)
    elif accion_id.startswith("color_"):
        estado["idx_color"] = int(accion_id.split("_")[1])

def verificar_modelo():
    if os.path.exists(RUTA_MODELO): return True
    import urllib.request
    print("  Descargando modelo...")
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    try:
        urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)
        return True
    except: return False

def iniciar_camara():
    cap = cv2.VideoCapture(CAMARA_ID)
    if not cap.isOpened(): return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS, FPS_OBJETIVO)
    for _ in range(15): cap.read()
    print("=" * 56)
    print("  TouchWall — Paso 8: Gestos + Botones (Integrados)")
    print("=" * 56)
    print("  👆 ÍNDICE   = Mover cursor (hover en botones)")
    print("  🤏 PINCH    = Clic (en botón) o Dibujar (en lienzo)")
    print("  ✌️ JUNTOS   = Borrador (Índice y Medio pegados)")
    print("  ✌️ VICTORIA = Cambiar color (Índice y Medio separados)")
    print("  ✊ PUÑO     = Pausa")
    print("=" * 56)
    return cap

def lerp(a, b, f): return a + (b - a) * f

def dibujar_esqueleto(frame, lm, alto, ancho):
    p = [(int(l.x * ancho), int(l.y * alto)) for l in lm]
    for c in CONEXIONES_MANO:
        if c.start < len(p) and c.end < len(p):
            cv2.line(frame, p[c.start], p[c.end], (80, 80, 80), 1)
    for i, (cx, cy) in enumerate(p):
        cv2.circle(frame, (cx, cy), 3, (120, 120, 120), -1)

def dibujar_cursor(frame, cx, cy, gesto, color_pincel):
    if gesto == GESTO_PINCH:
        cv2.circle(frame, (cx, cy), 16, color_pincel, 2)
        cv2.circle(frame, (cx, cy), 7, color_pincel, -1)
    elif gesto == GESTO_DOS_JUNTOS:
        cv2.circle(frame, (cx, cy), GROSOR_BORRADOR, (150, 150, 150), 2)
    elif gesto == GESTO_INDICE:
        cv2.circle(frame, (cx, cy), 10, (0, 220, 255), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 220, 255), -1)
    else:
        cv2.circle(frame, (cx, cy), 8, (100, 100, 100), 2)

def dibujar_hud(frame, fps, gesto, color_nom, grosor):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ALTO - 50), (ANCHO, ALTO), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    texto = f"FPS:{fps:.0f}  |  Gesto: {gesto}  |  Color: {color_nom}  |  Grosor: {grosor}px"
    cv2.putText(frame, texto, (BTN_X + BTN_W + 20, ALTO - 20), FUENTE, 0.6, (200, 200, 200), 1)
    cv2.circle(frame, (ANCHO - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame, "LIVE", (ANCHO - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)


# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    if not verificar_modelo(): return
    cap = iniciar_camara()
    if not cap: return

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=RUTA_MODELO),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    estabiliz = Estabilizador()
    botones = construir_botones()

    lienzo = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
    estado = {"idx_color": 0, "grosor": GROSOR_INICIAL}
    
    cx_suave = float(ANCHO // 2)
    cy_suave = float(ALTO // 2)
    cx_prev = cy_prev = None

    tiempo_inicio = cv2.getTickCount()
    contador_frames = 0
    fps = 0.0
    ts_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        ts_ms += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(mp_img, ts_ms)

        cx_int = cy_int = None
        gesto_final = GESTO_OTRO
        sobre_boton = False

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            cx_suave = lerp(cx_suave, lm[INDICE_LM].x * ANCHO, LERP_FACTOR)
            cy_suave = lerp(cy_suave, lm[INDICE_LM].y * ALTO, LERP_FACTOR)
            cx_int, cy_int = int(cx_suave), int(cy_suave)

            gesto_raw = clasificar_gesto(lm)
            gesto_final = estabiliz.actualizar(gesto_raw)
            
            pinch_nuevo = (gesto_final == GESTO_PINCH and estabiliz.nuevo)

            # 1) Disparo: VICTORIA (Cambiar color)
            if estabiliz.disparo and gesto_final == GESTO_VICTORIA:
                estado["idx_color"] = (estado["idx_color"] + 1) % len(PALETA_COLORES)

            # 2) Procesar botones
            for btn in botones:
                btn.actualizar(cx_int, cy_int, pinch_nuevo)
                if btn.contiene(cx_int, cy_int):
                    sobre_boton = True
                if btn.activo:
                    procesar_accion(btn.accion_id, estado, lienzo)

            # 3) Interacción con lienzo (solo si no estamos sobre un botón ni en pausa)
            if gesto_final != GESTO_PUÑO:
                if gesto_final == GESTO_PINCH and not sobre_boton:
                    if cx_prev is not None:
                        cv2.line(lienzo, (cx_prev, cy_prev), (cx_int, cy_int),
                                 PALETA_COLORES[estado["idx_color"]], estado["grosor"])
                    else:
                        cv2.circle(lienzo, (cx_int, cy_int), estado["grosor"]//2, 
                                   PALETA_COLORES[estado["idx_color"]], -1)
                    cx_prev, cy_prev = cx_int, cy_int
                
                elif gesto_final == GESTO_DOS_JUNTOS and not sobre_boton:
                    cv2.circle(lienzo, (cx_int, cy_int), GROSOR_BORRADOR, (0,0,0), -1)
                    cx_prev, cy_prev = cx_int, cy_int
                else:
                    cx_prev, cy_prev = None, None
            else:
                cx_prev, cy_prev = None, None

            dibujar_esqueleto(frame, lm, ALTO, ANCHO)
        else:
            estabiliz.actualizar(GESTO_OTRO)
            cx_prev, cy_prev = None, None
            for btn in botones: btn.actualizar(-1, -1, False)

        # ── Composición y UI ──
        mask = cv2.cvtColor(lienzo, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        idx = mask > 0
        frame[idx] = np.clip(lienzo[idx] * ALPHA_LIENZO + frame[idx] * (1 - ALPHA_LIENZO), 0, 255).astype(np.uint8)

        dibujar_fondo_barra(frame, len(botones))
        for btn in botones:
            override = tuple(max(20, c//3) for c in PALETA_COLORES[int(btn.accion_id.split("_")[1])]) if btn.accion_id.startswith("color_") else None
            btn.dibujar(frame, override)

        if cx_int is not None:
            dibujar_cursor(frame, cx_int, cy_int, gesto_final, PALETA_COLORES[estado["idx_color"]])

        if gesto_final == GESTO_PUÑO:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (ANCHO, ALTO), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            cv2.putText(frame, "PAUSADO ✊", (ANCHO//2 - 120, ALTO//2), FUENTE, 1.4, (100,100,100), 3)

        contador_frames += 1
        if contador_frames % 30 == 0:
            tf = cv2.getTickCount()
            fps = 30 / ((tf - tiempo_inicio) / cv2.getTickFrequency())
            tiempo_inicio = tf
            contador_frames = 0
            
        dibujar_hud(frame, fps, gesto_final, NOMBRES_COLORES[estado["idx_color"]], estado["grosor"])
        
        cv2.imshow(NOMBRE_VENTANA, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
