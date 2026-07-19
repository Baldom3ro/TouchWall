import subprocess
import cv2

def get_available_cameras():
    """
    Usa PowerShell para obtener los nombres reales de las cámaras en Windows,
    y luego verifica cuáles índices de OpenCV están disponibles realmente.
    Retorna un diccionario {indice: "Nombre de la Cámara"}.
    """
    nombres = []
    try:
        # Comando para sacar los nombres "Amigables" de las cámaras instaladas
        cmd = 'powershell "Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName"'
        output = subprocess.check_output(cmd, shell=True, text=True)
        nombres = [line.strip() for line in output.split('\n') if line.strip()]
    except Exception as e:
        print(f"[Error Scanner] No se pudieron obtener nombres de cámaras: {e}")

    # Si por alguna razón falla, damos nombres genéricos
    if not nombres:
        nombres = ["Cámara 0", "Cámara 1", "Cámara 2", "Cámara 3"]

    camaras_validas = {}
    
    # Probar los primeros índices para ver si abren
    for i in range(len(nombres)):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            nombre = nombres[i] if i < len(nombres) else f"Cámara USB {i}"
            camaras_validas[i] = nombre
            cap.release()
            
    # Si PowerShell no encontró nada pero OpenCV sí
    if not camaras_validas:
        for i in range(4):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                camaras_validas[i] = f"Cámara genérica {i}"
                cap.release()

    if not camaras_validas:
        camaras_validas[0] = "No se detectó cámara"

    return camaras_validas
