import cv2
import numpy as np

def calcular_matriz(puntos_camara, pantalla_w, pantalla_h):
    """
    Calcula la matriz de transformación de perspectiva (homografía).
    puntos_camara: Lista de 4 puntos [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    pts_src = np.array(puntos_camara, dtype=np.float32)
    # Las esquinas en la pantalla (destino)
    pts_dst = np.array([
        [0, 0],
        [pantalla_w, 0],
        [pantalla_w, pantalla_h],
        [0, pantalla_h]
    ], dtype=np.float32)
    
    matriz, status = cv2.findHomography(pts_src, pts_dst)
    return matriz

def aplicar_homografia(x, y, matriz):
    """
    Convierte coordenadas de cámara a coordenadas de pantalla usando la matriz H.
    """
    pt = np.array([[[x, y]]], dtype=np.float32)
    pt_trans = cv2.perspectiveTransform(pt, matriz)
    return pt_trans[0][0][0], pt_trans[0][0][1]
