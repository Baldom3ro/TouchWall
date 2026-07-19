"""
========================================================
  PASO 9 Y 10 - TouchWall (Final)
  Calibración por Homografía y Control del SO
========================================================
  Objetivo:
    Mapear la cámara a la superficie proyectada y
    controlar el ratón de Windows usando gestos.

  Modos:
    1. CALIBRACIÓN: Se muestran 4 puntos en las esquinas
       de la pantalla. Haz PINCH en cada uno.
    2. INTERACTIVO: El programa controla el ratón.

  Gestos en Modo Interactivo:
    👆 ÍNDICE   → Mueve el ratón sin hacer clic.
    🤏 PINCH    → Sostiene el clic izquierdo (permite arrastrar).
    ✊ PUÑO     → Pausa la cámara.

  Emergencia:
    - Failsafe: Arrastra el ratón a una esquina de tu
      monitor (con tu ratón físico) para forzar un aborto.
    - Teclado: Presiona 'q' en la ventana del monitor
      para cerrar el programa.
========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── CONFIGURACIÓN DE PYAUTOGUI ───
pyautogui.FAILSAFE = True      # Aborta si el mouse físico va a una esquina
pyautogui.PAUSE    = 0.0       # Evita lag interno al enviar comandos

# ─── CONFIGURACIÓN DE CÁMARA ───
CAMARA_ID      = 0
ANCHO          = 1920
ALTO           = 1080
FPS_OBJETIVO   = 30
NOMBRE_MONITOR = "TouchWall - Monitor"
NOMBRE_CALIB   = "TouchWall - Calibracion"

RUTA_MODELO = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "modelos", "hand_landmarker.task"
)

MAX_MANOS               = 2
MIN_CONFIANZA_DETECCION = 0.4
MIN_CONFIANZA_TRACKING  = 0.4
INDICE_LM               = 8

# ─── GESTOS Y ESTABILIZACIÓN (Escala Relativa) ───
# Estos umbrales ahora son un porcentaje del tamaño de tu mano (del 0.0 al 1.0)
UMBRAL_PINCH_INICIO = 0.15  # Pinch inicia al 15% del tamaño de la mano
UMBRAL_PINCH_FIN    = 0.25  # Pinch se suelta al 25% 
UMBRAL_SCROLL_INICIO = 0.12 # Scroll inicia al 12% 
UMBRAL_SCROLL_FIN    = 0.18 # Scroll se suelta al 18%
FRAMES_ESTABLE = 4

GESTO_INDICE = "INDICE"
GESTO_PINCH  = "PINCH"
GESTO_PUÑO   = "PUÑO"
GESTO_OTRO   = "OTRO"

FUENTE = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────────────
# MOTOR DE GESTOS
# ─────────────────────────────────────────────────────

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def _dedo_extendido(lm, punta, pip):
    return dist(lm[punta], lm[0]) > dist(lm[pip], lm[0])

def clasificar_gesto(lm) -> str:
    ext_indice  = _dedo_extendido(lm, 8,  6)
    ext_medio   = _dedo_extendido(lm, 12, 10)
    ext_anular  = _dedo_extendido(lm, 16, 14)
    ext_menique = _dedo_extendido(lm, 20, 18)
    
    escala_mano = dist(lm[0], lm[9])
    if escala_mano < 0.0001: escala_mano = 0.0001
    
    d_pinch = dist(lm[4], lm[8]) / escala_mano

    if d_pinch < UMBRAL_PINCH_INICIO:
        return GESTO_PINCH
    if not ext_indice and not ext_medio and not ext_anular and not ext_menique:
        return GESTO_PUÑO
    if ext_indice and not ext_medio and not ext_anular and not ext_menique:
        return GESTO_INDICE
    return GESTO_OTRO

class Estabilizador:
    def __init__(self, n=FRAMES_ESTABLE):
        self.n = n
        self.bruto = GESTO_OTRO
        self.contador = 0
        self.confirmado = GESTO_OTRO
        self.nuevo = False
        self.finalizado = False
    
    def actualizar(self, gesto_nuevo):
        self.nuevo = False
        self.finalizado = False
        
        if gesto_nuevo == self.bruto:
            self.contador = min(self.contador + 1, self.n)
        else:
            self.bruto = gesto_nuevo
            self.contador = 1
        
        nuevo_conf = self.bruto if self.contador >= self.n else self.confirmado
        
        if nuevo_conf != self.confirmado:
            if self.confirmado != GESTO_OTRO:
                self.finalizado = True  # Gesto anterior terminó
            self.confirmado = nuevo_conf
            self.nuevo = True
            
        return self.confirmado

# ─────────────────────────────────────────────────────
# UI DE CALIBRACIÓN
# ─────────────────────────────────────────────────────

def dibujar_pantalla_calibracion(canvas, pantalla_w, pantalla_h, puntos_capturados):
    """
    Dibuja una imagen a pantalla completa con indicadores en las esquinas.
    Devuelve la imagen (canvas) y las coordenadas esperadas (puntos de pantalla).
    """
    canvas.fill(0)
    margen = 80
    
    # 4 Esquinas destino: TL, TR, BR, BL
    esquinas = [
        (margen, margen),
        (pantalla_w - margen, margen),
        (pantalla_w - margen, pantalla_h - margen),
        (margen, pantalla_h - margen)
    ]
    
    # Textos centrales
    cv2.putText(canvas, "CALIBRACION DE PROYECTOR", (pantalla_w//2 - 350, pantalla_h//2 - 40), 
                FUENTE, 1.5, (255, 255, 255), 3)
    cv2.putText(canvas, f"Paso {puntos_capturados + 1} de 4", (pantalla_w//2 - 150, pantalla_h//2 + 20), 
                FUENTE, 1.2, (0, 255, 255), 2)
    cv2.putText(canvas, "Apunta al circulo resaltado y haz PINCH", (pantalla_w//2 - 350, pantalla_h//2 + 80), 
                FUENTE, 1.0, (200, 200, 200), 2)
    
    for i, pt in enumerate(esquinas):
        if i < puntos_capturados:
            # Ya capturado
            cv2.circle(canvas, pt, 30, (0, 255, 0), -1)
            cv2.putText(canvas, "OK", (pt[0]-15, pt[1]+10), FUENTE, 1, (0,0,0), 2)
        elif i == puntos_capturados:
            # Objetivo actual
            cv2.circle(canvas, pt, 40, (0, 0, 255), -1)
            cv2.circle(canvas, pt, 20, (255, 255, 255), -1)
        else:
            # Pendiente
            cv2.circle(canvas, pt, 15, (100, 100, 100), -1)
            
    return canvas, esquinas

def aplicar_homografia(cx, cy, H):
    """Convierte el punto de la cámara al punto de la pantalla usando H."""
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    pt_trans = cv2.perspectiveTransform(pt, H)
    return pt_trans[0][0][0], pt_trans[0][0][1]

def lerp(a, b, f): return a + (b - a) * f

# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    pantalla_w, pantalla_h = pyautogui.size()
    print(f"  [INFO] Resolución del monitor: {pantalla_w}x{pantalla_h}")
    
    cap = cv2.VideoCapture(CAMARA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS, FPS_OBJETIVO)
    
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=RUTA_MODELO),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=MAX_MANOS,
        min_hand_detection_confidence=MIN_CONFIANZA_DETECCION,
        min_hand_presence_confidence=MIN_CONFIANZA_TRACKING,
        min_tracking_confidence=MIN_CONFIANZA_TRACKING
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    estabiliz = Estabilizador()
    
    # Estado
    modo = "CALIBRACION"
    puntos_camara = []
    puntos_pantalla = []
    matriz_H = None
    
    cx_suave, cy_suave = ANCHO/2, ALTO/2
    wrist_suave_x, wrist_suave_y = ANCHO/2, ALTO/2
    ts_ms = 0
    
    # Variables de histeresis y compensación
    offset_x = 0
    offset_y = 0
    mouse_presionado = False
    right_click_presionado = False
    scroll_activo = False
    scroll_ref_y = 0
    zoom_presionado = False
    zoom_ref = 0
    
    # Variables de UI
    canvas_calibracion = np.zeros((pantalla_h, pantalla_w, 3), dtype=np.uint8)
    
    # Ventana de calibración a pantalla completa
    cv2.namedWindow(NOMBRE_CALIB, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(NOMBRE_CALIB, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 60)
    print("  MODO CALIBRACIÓN")
    print("  Apunta a las 4 esquinas marcadas y junta tus dedos (PINCH)")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("  [ADVERTENCIA] Frame vacío o cámara desconectada.")
            time.sleep(0.1)
            continue
            
        frame = frame.copy()  # EVITA CORRUPCIÓN DE MEMORIA Y ACCESS VIOLATIONS EN WINDOWS

        ts_ms += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(mp_img, ts_ms)

        gesto_final = GESTO_OTRO
        cx_int = cy_int = None
        pinch_iniciado = False
        pinch_soltado  = False
        right_click_iniciado = False
        
        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            # ── Suavizado Dinámico Inteligente (Filtro 1€ Simplificado) ──
            # Se mueve rápido = muy responsivo. Se mueve lento = muy estable (cero temblor)
            
            # Cursor normal (Índice)
            idx_x = lm[INDICE_LM].x * ANCHO
            idx_y = lm[INDICE_LM].y * ALTO
            dist_idx = math.hypot(idx_x - cx_suave, idx_y - cy_suave)
            lerp_idx = 0.05 + (min(dist_idx, 80) / 80.0) * 0.4
            cx_suave = cx_suave + (idx_x - cx_suave) * lerp_idx
            cy_suave = cy_suave + (idx_y - cy_suave) * lerp_idx
            
            # Ancla de arrastre (Muñeca)
            w_x = lm[0].x * ANCHO
            w_y = lm[0].y * ALTO
            dist_w = math.hypot(w_x - wrist_suave_x, w_y - wrist_suave_y)
            lerp_w = 0.05 + (min(dist_w, 80) / 80.0) * 0.4
            wrist_suave_x = wrist_suave_x + (w_x - wrist_suave_x) * lerp_w
            wrist_suave_y = wrist_suave_y + (w_y - wrist_suave_y) * lerp_w
            
            gesto_raw = clasificar_gesto(lm)
            gesto_final = estabiliz.actualizar(gesto_raw)
            
            # ── LÓGICA DE GESTOS (Independiente de la distancia) ──
            escala_mano = dist(lm[0], lm[9])
            if escala_mano < 0.0001: escala_mano = 0.0001
            
            d_pinch = dist(lm[4], lm[8]) / escala_mano
            d_right = dist(lm[4], lm[20]) / escala_mano
            d_scroll = dist(lm[8], lm[12]) / escala_mano
            
            # ── ZOOM A 2 MANOS (Pellizco Doble) ──
            zoom_activo = False
            zoom_delta = 0
            if len(res.hand_landmarks) == 2:
                lm2 = res.hand_landmarks[1]
                escala2 = max(0.0001, dist(lm2[0], lm2[9]))
                d_pinch2 = dist(lm2[4], lm2[8]) / escala2
                
                if d_pinch < UMBRAL_PINCH_INICIO and d_pinch2 < UMBRAL_PINCH_INICIO:
                    zoom_activo = True
                    dist_manos = dist(lm[8], lm2[8]) * ANCHO
                    if not zoom_presionado:
                        zoom_presionado = True
                        zoom_ref = dist_manos
                    else:
                        delta_z = dist_manos - zoom_ref
                        if delta_z > 40:
                            pasos = int(delta_z // 40)
                            zoom_delta = 120 * pasos
                            zoom_ref = dist_manos
                        elif delta_z < -40:
                            pasos = int(abs(delta_z) // 40)
                            zoom_delta = -120 * pasos
                            zoom_ref = dist_manos
                else:
                    zoom_presionado = False
            else:
                zoom_presionado = False
                
            # Clic Izquierdo (PINCH)
            if zoom_activo:
                if mouse_presionado:
                    mouse_presionado = False
                    pinch_soltado = True # Liberar clic si empezamos a hacer zoom
            else:
                if not mouse_presionado and d_pinch < UMBRAL_PINCH_INICIO:
                    mouse_presionado = True
                    pinch_iniciado = True
                elif mouse_presionado and d_pinch > UMBRAL_PINCH_FIN:
                    mouse_presionado = False
                    pinch_soltado = True

            # Clic Derecho (Pulgar + Meñique)
            if not right_click_presionado and d_right < UMBRAL_PINCH_INICIO:
                right_click_presionado = True
                right_click_iniciado = True
            elif right_click_presionado and d_right > UMBRAL_PINCH_FIN:
                right_click_presionado = False

            # Scroll (Índice + Medio unidos) con histéresis
            if gesto_final != GESTO_PUÑO:
                if not scroll_activo and d_scroll < UMBRAL_SCROLL_INICIO:
                    scroll_activo = True
                    scroll_ref_y = wrist_suave_y
                elif scroll_activo and d_scroll > UMBRAL_SCROLL_FIN:
                    scroll_activo = False
            else:
                scroll_activo = False

            # ── COMPENSACIÓN DE CAÍDA (DRAG) ──
            if mouse_presionado and not scroll_activo and not zoom_activo:
                if pinch_iniciado:
                    # En el milisegundo exacto que inicia el pinch, guardamos la distancia
                    # entre el cursor actual y la muñeca.
                    offset_x = cx_suave - wrist_suave_x
                    offset_y = cy_suave - wrist_suave_y
                
                # Durante el arrastre, el cursor es movido SOLO por la muñeca
                cx_int = int(wrist_suave_x + offset_x)
                cy_int = int(wrist_suave_y + offset_y)
            else:
                # Movimiento libre normal (incluso si está scrolleando, para dibujar el HUD)
                offset_x = 0
                offset_y = 0
                cx_int = int(cx_suave)
                cy_int = int(cy_suave)
            
            # Dibujar un pequeño punto en la cámara
            cv2.circle(frame, (cx_int, cy_int), 5, (0, 255, 255), -1)

        else:
            estabiliz.actualizar(GESTO_OTRO)
            
        # ──────────────────────────────────────────────────────────
        # MODO: CALIBRACIÓN
        # ──────────────────────────────────────────────────────────
        if modo == "CALIBRACION":
            canvas, esquinas_esperadas = dibujar_pantalla_calibracion(canvas_calibracion, pantalla_w, pantalla_h, len(puntos_camara))
            
            if pinch_iniciado and cx_int is not None:
                puntos_camara.append([cx_int, cy_int])
                print(f"  [OK] Esquina {len(puntos_camara)} capturada en: ({cx_int}, {cy_int})")
                
                # Feedback visual temporal
                cv2.circle(canvas, esquinas_esperadas[len(puntos_camara)-1], 50, (255, 255, 255), -1)
                
                if len(puntos_camara) == 4:
                    # Calcular homografía
                    pts_src = np.array(puntos_camara, dtype=np.float32)
                    pts_dst = np.array(esquinas_esperadas, dtype=np.float32)
                    matriz_H, _ = cv2.findHomography(pts_src, pts_dst)
                    
                    print("\n  [EXITO] Calibración terminada.")
                    print("  Matriz calculada:\n", matriz_H)
                    print("\n  PASANDO A MODO CONTROL DEL SISTEMA...")
                    print("  Puedes salir presionando 'q' en la ventana del monitor.")
                    
                    cv2.destroyWindow(NOMBRE_CALIB)
                    modo = "ACTIVO"
                    
            if modo == "CALIBRACION":
                cv2.imshow(NOMBRE_CALIB, canvas)
                
        # ──────────────────────────────────────────────────────────
        # MODO: ACTIVO (Control del Mouse)
        # ──────────────────────────────────────────────────────────
        elif modo == "ACTIVO":
            if gesto_final != GESTO_PUÑO and cx_int is not None:
                
                if zoom_activo:
                    # Lógica de Zoom (Ctrl + Rueda)
                    if zoom_delta != 0:
                        pyautogui.keyDown('ctrl')
                        time.sleep(0.02) # Darle tiempo a Windows para registrar la tecla Ctrl
                        pyautogui.scroll(zoom_delta)
                        pyautogui.keyUp('ctrl')
                        print(f"  [ZOOM] {'Acercar' if zoom_delta > 0 else 'Alejar'} ({zoom_delta})")
                elif scroll_activo:
                    # Lógica de Scroll Vertical
                    dy = wrist_suave_y - scroll_ref_y
                    
                    # En Windows, un "clic" de rueda de ratón estándar es de 120 unidades
                    if dy < -12:  # Mano sube -> Scroll arriba
                        pyautogui.scroll(120)
                        scroll_ref_y = wrist_suave_y
                        print("  [SCROLL] Arriba")
                    elif dy > 12: # Mano baja -> Scroll abajo
                        pyautogui.scroll(-120)
                        scroll_ref_y = wrist_suave_y
                        print("  [SCROLL] Abajo")
                else:
                    # 1. Transformar coordenada de cámara a coordenada de pantalla (Windows)
                    sx_float, sy_float = aplicar_homografia(cx_int, cy_int, matriz_H)
                    
                    # 2. Clípping de seguridad (Failsafe margin)
                    sx = int(max(2, min(pantalla_w - 2, sx_float)))
                    sy = int(max(2, min(pantalla_h - 2, sy_float)))
                    
                    # 3. Mover el ratón del SO
                    # Evitar tocar exactamente los bordes (0 o max) para no disparar el FailSafe automático
                    sx_safe = max(5, min(pantalla_w - 5, int(sx)))
                    sy_safe = max(5, min(pantalla_h - 5, int(sy)))
                    pyautogui.moveTo(sx_safe, sy_safe, _pause=False)
                    
                    # 4. Manejar los Clics
                    if right_click_iniciado:
                        pyautogui.click(button='right')
                        print("  [CLIC] Right Click")
                        
                    if pinch_iniciado:
                        pyautogui.mouseDown(button='left')
                        print("  [CLIC] mouseDown")
                    elif pinch_soltado:
                        pyautogui.mouseUp(button='left')
                        print("  [CLIC] mouseUp")
                        
            # Si se interrumpe la visión, soltar clic por seguridad
            if gesto_final == GESTO_OTRO and mouse_presionado:
                pyautogui.mouseUp(button='left')
                mouse_presionado = False

            # Mostrar monitor pequeño para que el usuario no se sienta ciego
            frame_monitor = cv2.resize(frame, (320, 180))
            if zoom_activo:
                cv2.putText(frame_monitor, "ZOOM ACTIVO", (10, 30), FUENTE, 0.7, (0, 255, 255), 2)
            elif mouse_presionado:
                cv2.rectangle(frame_monitor, (0,0), (319, 179), (0, 200, 0), 4)
            if gesto_final == GESTO_PUÑO:
                cv2.putText(frame_monitor, "PAUSA", (100, 90), FUENTE, 1.0, (0, 0, 255), 2)
                
            cv2.imshow(NOMBRE_MONITOR, frame_monitor)
            
        # ── Limpieza de memoria (Previene Fuga de RAM) ──
        del frame, rgb, mp_img, res
        if 'frame_monitor' in locals():
            del frame_monitor
            
        # ── Teclado ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n  Cerrando TouchWall...")
            break

    if modo == "ACTIVO" and mouse_presionado:
        pyautogui.mouseUp()
        
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
