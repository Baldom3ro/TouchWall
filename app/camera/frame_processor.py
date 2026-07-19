import cv2
import time
import threading
import json
import os
import numpy as np

from app.config.constants import *
from app.camera.camera_manager import CameraManager
from app.tracking.hand_detector import HandDetector
from app.tracking.smoothing import Estabilizador
from app.system.mouse_controller import MouseController
from app.gestures.gesture_manager import clasificar_gesto, dist
from app.projection.homography import calcular_matriz, aplicar_homografia
from app.config.settings_manager import load_settings

class TouchWallEngine:
    """
    El motor principal de TouchWall.
    Une la cámara, la IA, el suavizado y el ratón en un bucle que corre
    en un hilo independiente a la interfaz gráfica.
    """
    def __init__(self, ui_callback=None):
        self.settings = load_settings()
        
        nombre_cam = self.settings.get("camara", {}).get("nombre", "")
        cam_id = CAMARA_ID
        if nombre_cam and "-" in nombre_cam:
            try:
                cam_id = int(nombre_cam.split("-")[0].strip())
            except: pass
            
        self.camera = CameraManager(camera_id=cam_id, width=ANCHO, height=ALTO)
        self.detector = HandDetector()
        self.mouse = MouseController()
        
        self.estabilizador = Estabilizador(alpha_min=0.05, alpha_max=0.8, umbral_velocidad=500)
        self.estabilizador_muñeca = Estabilizador(alpha_min=0.1, alpha_max=0.8, umbral_velocidad=500)
        self.load_dynamic_settings()
        
        self.is_running = False
        self.thread = None
        self.modo = "INACTIVO" # "INACTIVO", "CALIBRACION", "ACTIVO"
        
        # Estado Interno
        self.matriz_h = None
        self.puntos_calibracion = []
        self.canvas_calibracion = np.zeros((ALTO, ANCHO, 3), dtype=np.uint8)
        
        # Variables de Gestos
        self.mouse_presionado = False
        self.right_click_presionado = False
        self.double_click_presionado = False
        self.zoom_presionado = False
        self.zoom_ref = 0
        self.scroll_activo = False
        self.scroll_ref_y = 0
        self.ts_ms = 0
        
        # Stats para Dashboard
        self.mano_detectada = False
        self.fps = 0
        self._fps_frames = 0
        self._fps_t0 = time.time()
        self.tiempo_inicio = 0
        
        self.ui_callback = ui_callback # Para enviar mensajes a la UI

    def load_dynamic_settings(self):
        # Mapeo de sliders (0-100) a matemáticas del filtro
        # Sensibilidad: Menos suavizado base. (0 = 0.01, 100 = 0.5)
        # Suavizado: Límite superior del suavizado dinámico (0 = 0.1, 100 = 1.0)
        # Velocidad: Qué tan rápido responde a movs rápidos (0 = 1000px/s, 100 = 50px/s)
        sensibilidad = self.settings["avanzado"].get("Sensibilidad", 50)
        suavizado = self.settings["avanzado"].get("Suavizado", 50)
        velocidad = self.settings["avanzado"].get("Velocidad", 50)
        
        self.estabilizador.alpha_min = 0.01 + (sensibilidad / 100.0) * 0.49
        self.estabilizador.alpha_max = 0.1 + (suavizado / 100.0) * 0.9
        self.estabilizador.umbral_velocidad = 1000 - (velocidad / 100.0) * 950
        
        self.estabilizador_muñeca.alpha_min = self.estabilizador.alpha_min
        self.estabilizador_muñeca.alpha_max = self.estabilizador.alpha_max
        self.estabilizador_muñeca.umbral_velocidad = self.estabilizador.umbral_velocidad

    def get_stats(self):
        return {
            "fps": self.fps,
            "mano_detectada": self.mano_detectada,
            "matriz_lista": self.matriz_h is not None,
            "is_running": self.is_running,
            "modo": self.modo,
            "tiempo_activo": time.time() - self.tiempo_inicio if self.is_running else 0
        }

    def start(self):
        if self.is_running: return
        self.camera.start()
        
        # Cargar matriz de calibración si existe
        if os.path.exists(CALIBRATION_FILE):
            try:
                with open(CALIBRATION_FILE, 'r') as f:
                    data = json.load(f)
                    self.matriz_h = np.array(data["matriz"])
                    self.modo = "ACTIVO"
                    self._log("Configuración cargada. Modo ACTIVO.")
            except Exception as e:
                self._log("Error cargando calibración previa.")
                self.modo = "CALIBRACION"
        else:
            self.modo = "CALIBRACION"

        self.tiempo_inicio = time.time()
        self.is_running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.is_running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=1.0)
        self.camera.stop()
        cv2.destroyAllWindows()
        self._log("Motor detenido.")

    def iniciar_calibracion(self):
        self.puntos_calibracion = []
        self.modo = "CALIBRACION"
        self.matriz_h = None
        self._log("Modo Calibración iniciado. Pellizca en las 4 esquinas.")

    def _log(self, mensaje):
        print(f"[Engine] {mensaje}")
        if self.ui_callback:
            self.ui_callback(mensaje)

    def _dibujar_calibracion(self):
        self.canvas_calibracion.fill(0)
        margen = 80
        esquinas = [
            (margen, margen), (ANCHO - margen, margen),
            (ANCHO - margen, ALTO - margen), (margen, ALTO - margen)
        ]
        
        actual = len(self.puntos_calibracion)
        for i, pt in enumerate(esquinas):
            color = (0, 255, 0) if i < actual else (0, 0, 255) if i == actual else (100, 100, 100)
            grosor = -1 if i < actual else 4
            cv2.circle(self.canvas_calibracion, pt, 20, color, grosor)
            if i == actual:
                cv2.circle(self.canvas_calibracion, pt, 30, (255, 255, 255), 2)
                
    def _loop(self):
        while self.is_running:
            ret, frame = self.camera.read(block=True)
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            # Cálculo de FPS
            self._fps_frames += 1
            t_ahora = time.time()
            if t_ahora - self._fps_t0 >= 1.0:
                self.fps = self._fps_frames
                self._fps_frames = 0
                self._fps_t0 = t_ahora
                
            self.ts_ms += 33
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ml = cv2.resize(frame_rgb, (640, 360))
            res = self.detector.detect(frame_ml, self.ts_ms)
            
            self.mano_detectada = bool(res and res.hand_landmarks)
            
            # Variables de este frame
            gesto_final = GESTO_OTRO
            pinch_iniciado = False
            pinch_soltado = False
            right_click_iniciado = False
            double_click_iniciado = False
            zoom_activo = False
            zoom_delta = 0
            
            cx_int = cy_int = None

            if res and res.hand_landmarks:
                lm = res.hand_landmarks[0]
                
                # Coordenadas crudas
                cx_crudo = int(lm[INDICE_LM].x * ANCHO)
                cy_crudo = int(lm[INDICE_LM].y * ALTO)
                
                # Suavizado Cursor Principal
                cx_float, cy_float = self.estabilizador.actualizar(cx_crudo, cy_crudo)
                cx_int, cy_int = int(cx_float), int(cy_float)
                
                # Suavizado Muñeca (Para Scroll)
                wrist_crudo_y = int(lm[0].y * ALTO)
                _, wrist_suave_y = self.estabilizador_muñeca.actualizar(0, wrist_crudo_y)
                
                # Cálculo de Pinch y Escala
                escala_mano = max(0.0001, dist(lm[0], lm[9]))
                d_pinch = dist(lm[PULGAR_LM], lm[INDICE_LM]) / escala_mano
                d_double = dist(lm[PULGAR_LM], lm[MEDIO_LM]) / escala_mano
                d_right = dist(lm[PULGAR_LM], lm[MENIQUE_LM]) / escala_mano
                d_index_middle = dist(lm[INDICE_LM], lm[MEDIO_LM]) / escala_mano
                
                gesto_final = clasificar_gesto(lm)
                
                # Scroll Vertical (Índice y Medio unidos)
                if gesto_final != GESTO_PUÑO:
                    if d_index_middle < 0.4 and not self.zoom_presionado:
                        if not self.scroll_activo:
                            self.scroll_activo = True
                            self.scroll_ref_y = wrist_suave_y
                    else:
                        self.scroll_activo = False
                else:
                    self.scroll_activo = False

                # Manejo de Multitouch (Zoom)
                if len(res.hand_landmarks) == 2:
                    lm2 = res.hand_landmarks[1]
                    escala2 = max(0.0001, dist(lm2[0], lm2[9]))
                    d_pinch2 = dist(lm2[PULGAR_LM], lm2[INDICE_LM]) / escala2
                    
                    if d_pinch < UMBRAL_PINCH_INICIO and d_pinch2 < UMBRAL_PINCH_INICIO:
                        zoom_activo = True
                        dist_manos = dist(lm[INDICE_LM], lm2[INDICE_LM]) * ANCHO
                        if not self.zoom_presionado:
                            self.zoom_presionado = True
                            self.zoom_ref = dist_manos
                        else:
                            delta_z = dist_manos - self.zoom_ref
                            if delta_z > 40:
                                pasos = int(delta_z // 40)
                                zoom_delta = 120 * pasos
                                self.zoom_ref = dist_manos
                            elif delta_z < -40:
                                pasos = int(abs(delta_z) // 40)
                                zoom_delta = -120 * pasos
                                self.zoom_ref = dist_manos
                    else:
                        self.zoom_presionado = False
                else:
                    self.zoom_presionado = False
                    
                # Clic Izquierdo
                if zoom_activo:
                    if self.mouse_presionado:
                        self.mouse_presionado = False
                        pinch_soltado = True
                else:
                    if not self.mouse_presionado and d_pinch < UMBRAL_PINCH_INICIO:
                        self.mouse_presionado = True
                        pinch_iniciado = True
                    elif self.mouse_presionado and d_pinch > UMBRAL_PINCH_FIN:
                        self.mouse_presionado = False
                        pinch_soltado = True
                        
                # Clic Derecho
                if not self.right_click_presionado and d_right < UMBRAL_PINCH_INICIO:
                    self.right_click_presionado = True
                    right_click_iniciado = True
                elif self.right_click_presionado and d_right > UMBRAL_PINCH_FIN:
                    self.right_click_presionado = False
                    
                # Doble Clic
                if not self.double_click_presionado and d_double < UMBRAL_PINCH_INICIO:
                    self.double_click_presionado = True
                    double_click_iniciado = True
                elif self.double_click_presionado and d_double > UMBRAL_PINCH_FIN:
                    self.double_click_presionado = False

            # ─── LÓGICA DE ESTADOS ───
            if self.modo == "CALIBRACION":
                self._dibujar_calibracion()
                if pinch_iniciado and cx_int is not None:
                    self.puntos_calibracion.append([cx_int, cy_int])
                    self._log(f"Punto {len(self.puntos_calibracion)} capturado.")
                    
                    if len(self.puntos_calibracion) == 4:
                        self.matriz_h = calcular_matriz(self.puntos_calibracion, ANCHO, ALTO)
                        self.modo = "ACTIVO"
                        cv2.destroyWindow(NOMBRE_CALIB)
                        self._log("Calibración completada.")
                        # Guardar matriz
                        try:
                            with open(CALIBRATION_FILE, 'w') as f:
                                json.dump({"matriz": self.matriz_h.tolist()}, f)
                        except: pass
                
                # ─── MODO CALIBRACIÓN ───
            if self.modo == "CALIBRACION":
                import pyautogui
                sw, sh = pyautogui.size()
                # Forzar que la imagen llene el 100% del monitor/proyector
                calib_img = cv2.resize(self.canvas_calibracion, (sw, sh))
                
                cv2.imshow(NOMBRE_CALIB, calib_img)
                # Forzar fullscreen solo la primera vez que se abre
                if not hasattr(self, 'calib_window_created'):
                    self.calib_window_created = True
                    cv2.setWindowProperty(NOMBRE_CALIB, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            elif self.modo == "ACTIVO" and self.matriz_h is not None:
                gestos_conf = self.settings.get("gestos", {})
                
                if gesto_final != GESTO_PUÑO and cx_int is not None:
                    sx, sy = aplicar_homografia(cx_int, cy_int, self.matriz_h)
                    if gestos_conf.get("Mover cursor", True):
                        self.mouse.mover(sx, sy)
                    
                    if zoom_activo and zoom_delta != 0 and gestos_conf.get("Zoom", False):
                        self.mouse.zoom(zoom_delta)
                    elif self.scroll_activo and gestos_conf.get("Scroll", True):
                        dy = wrist_suave_y - self.scroll_ref_y
                        if abs(dy) > 10:
                            if dy > 0: self.mouse.scroll_vertical(-80)
                            else: self.mouse.scroll_vertical(80)
                            self.scroll_ref_y = wrist_suave_y
                    
                    if right_click_iniciado and gestos_conf.get("Click", True):
                        self.mouse.click_derecho()
                        
                    if double_click_iniciado and gestos_conf.get("Doble click", True):
                        self.mouse.click_izquierdo()
                        self.mouse.click_izquierdo()
                        
                    if pinch_iniciado and gestos_conf.get("Click", True):
                        self.mouse.mouse_down()
                    elif pinch_soltado and gestos_conf.get("Click", True):
                        self.mouse.mouse_up()

            # ─── MONITOR DE CÁMARA ───
            frame_monitor = cv2.resize(frame, (320, 180))
            if zoom_activo:
                cv2.putText(frame_monitor, "ZOOM ACTIVO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif self.mouse_presionado:
                cv2.rectangle(frame_monitor, (0,0), (319, 179), (0, 200, 0), 4)
            if gesto_final == GESTO_PUÑO:
                cv2.putText(frame_monitor, "PAUSA", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
            cv2.imshow(NOMBRE_MONITOR, frame_monitor)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Fuera del bucle
        self.stop()
