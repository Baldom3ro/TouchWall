"""
========================================================
  PASO 1 - Iniciar Cámara
  Proyecto: Touch Wall (Proyección Interactiva)
========================================================
  Objetivo:
    Abrir la webcam, capturar el flujo de video en
    tiempo real y mostrarlo en pantalla.
    
  Controles:
    - Presiona 'q' para salir
    - Presiona 's' para tomar una captura de pantalla

  Librerías requeridas:
    pip install opencv-python
========================================================
"""

import cv2
import os
from datetime import datetime

# ─────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────

CAMARA_ID       = 0          # 0 = webcam integrada, 1 = webcam externa
ANCHO           = 1280       # Resolución de captura (ancho)
ALTO            = 720        # Resolución de captura (alto)
FPS_OBJETIVO    = 30         # Frames por segundo solicitados a la cámara
NOMBRE_VENTANA  = "TouchWall - Paso 1: Iniciar Camara"

COLOR_INFO      = (0, 255, 0)       # Verde para el texto de FPS
COLOR_AVISO     = (0, 100, 255)     # Naranja para avisos
GROSOR_TEXTO    = 2
FUENTE          = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────────────
# INICIALIZAR CÁMARA
# ─────────────────────────────────────────────────────

def iniciar_camara():
    """Abre la cámara y configura su resolución y FPS."""
    cap = cv2.VideoCapture(CAMARA_ID)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara con ID={CAMARA_ID}")
        print("        Verifica que tu webcam esté conectada y no esté en uso por otro programa.")
        return None

    # Aplicar configuración de resolución y FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS,          FPS_OBJETIVO)

    # Leer los valores reales que aceptó el driver de la cámara
    ancho_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_real  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_real   = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 50)
    print("  TouchWall — Paso 1: Iniciar Cámara")
    print("=" * 50)
    print(f"  Cámara ID  : {CAMARA_ID}")
    print(f"  Resolución : {ancho_real} x {alto_real} px")
    print(f"  FPS        : {fps_real}")
    print("=" * 50)
    print("  [q] Salir    [s] Captura de pantalla")
    print("=" * 50)

    return cap


# ─────────────────────────────────────────────────────
# DIBUJAR INFORMACIÓN EN PANTALLA (HUD)
# ─────────────────────────────────────────────────────

def dibujar_hud(frame, fps_actual):
    """Dibuja el HUD (Heads-Up Display) con información en tiempo real."""
    alto, ancho = frame.shape[:2]

    # — Fondo semitransparente para el panel de info superior —
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (350, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # — FPS —
    cv2.putText(frame,
                f"FPS: {fps_actual:.1f}",
                (10, 30), FUENTE, 0.8, COLOR_INFO, GROSOR_TEXTO)

    # — Resolución —
    cv2.putText(frame,
                f"Resolucion: {ancho} x {alto}",
                (10, 58), FUENTE, 0.65, COLOR_INFO, 1)

    # — Mensaje de controles en la esquina inferior izquierda —
    cv2.putText(frame,
                "[q] Salir  [s] Captura",
                (10, alto - 15), FUENTE, 0.55, (180, 180, 180), 1)

    # — Indicador LIVE en la esquina superior derecha —
    cv2.circle(frame, (ancho - 30, 25), 8, (0, 0, 255), -1)
    cv2.putText(frame,
                "LIVE",
                (ancho - 70, 32), FUENTE, 0.55, (0, 0, 255), 1)


# ─────────────────────────────────────────────────────
# GUARDAR CAPTURA
# ─────────────────────────────────────────────────────

def guardar_captura(frame):
    """Guarda el frame actual como imagen PNG en la carpeta 'capturas/'."""
    carpeta = "capturas"
    os.makedirs(carpeta, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = os.path.join(carpeta, f"captura_{timestamp}.png")

    cv2.imwrite(nombre_archivo, frame)
    print(f"  [OK] Captura guardada: {nombre_archivo}")


# ─────────────────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────────────────

def main():
    cap = iniciar_camara()
    if cap is None:
        return

    # Para calcular FPS reales
    contador_frames = 0
    fps_mostrar     = 0.0
    tiempo_inicio   = cv2.getTickCount()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[AVISO] No se pudo leer el frame. Verifica la cámara.")
            break

        # ── Calcular FPS real cada 30 frames ──
        contador_frames += 1
        if contador_frames % 30 == 0:
            tiempo_fin  = cv2.getTickCount()
            tiempo_seg  = (tiempo_fin - tiempo_inicio) / cv2.getTickFrequency()
            fps_mostrar = 30 / tiempo_seg
            tiempo_inicio = cv2.getTickCount()
            contador_frames = 0

        # ── Dibujar HUD sobre el frame ──
        dibujar_hud(frame, fps_mostrar)

        # ── Mostrar en ventana ──
        cv2.imshow(NOMBRE_VENTANA, frame)

        # ── Controles de teclado ──
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            print("\n  Cerrando TouchWall... Hasta luego!")
            break
        elif tecla == ord('s'):
            guardar_captura(frame)

    # ── Liberar recursos ──
    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
