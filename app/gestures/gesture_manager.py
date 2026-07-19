import math
from app.config.constants import *

def dist(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

def clasificar_gesto(lm):
    """
    Clasifica si la mano está haciendo puño cerrado (PAUSA).
    """
    if len(lm) < 21: return GESTO_OTRO
    
    dedos_cerrados = 0
    puntas = [8, 12, 16, 20]
    
    for punta in puntas:
        if lm[punta].y > lm[punta - 2].y:
            dedos_cerrados += 1
            
    if dedos_cerrados >= 3:
        return GESTO_PUÑO
    
    return GESTO_OTRO
