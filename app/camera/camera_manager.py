import cv2
import threading
import time

class CameraManager:
    """
    Se encarga de manejar el dispositivo físico de la cámara en un hilo separado.
    Esto evita que la interfaz gráfica (UI) se congele.
    """
    def __init__(self, camera_id=0, width=1920, height=1080):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        
        self.cap = None
        self.is_running = False
        
        self.current_frame = None
        self.lock = threading.Lock()
        self.frame_ready = threading.Event()
        self.thread = None

    def start(self):
        """Abre la cámara e inicia el hilo de captura."""
        if self.is_running:
            return

        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        # IMPORTANTE: Forzar MJPG para evitar el límite de 10 FPS del formato YUY2 por ancho de banda USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.cap.isOpened():
            print(f"[Error] No se pudo abrir la cámara ID: {self.camera_id}")
            return False

        self.is_running = True
        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()
        print("[Info] Cámara iniciada correctamente.")
        return True

    def _update_frame(self):
        """Bucle interno del hilo que captura los fotogramas constantemente."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.current_frame = frame.copy()
                self.frame_ready.set()
            else:
                time.sleep(0.01)

    def read(self, block=False):
        """
        Devuelve el fotograma. Si block=True, espera a que haya un fotograma nuevo.
        """
        if block:
            self.frame_ready.wait(timeout=1.0)
            
        with self.lock:
            if self.current_frame is not None:
                frame = self.current_frame.copy()
                self.frame_ready.clear() # Consumido
                return True, frame
            return False, None

    def stop(self):
        """Detiene el hilo y libera la cámara."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.current_frame = None
            
        print("[Info] Cámara detenida.")
