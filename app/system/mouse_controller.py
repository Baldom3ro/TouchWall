import pyautogui
from app.config.constants import *

class MouseController:
    """
    Se encarga de enviar todos los comandos al Sistema Operativo 
    usando PyAutoGUI (movers, clics, scroll).
    """
    def __init__(self):
        pyautogui.FAILSAFE = True      # Aborta si el mouse físico va a una esquina extrema
        pyautogui.PAUSE    = 0.0       # Evita lag interno al enviar comandos

    def mover(self, sx, sy):
        # Evitar tocar exactamente los bordes (0 o max) para no disparar el FailSafe automático de Windows
        screen_w, screen_h = pyautogui.size()
        sx_safe = max(5, min(screen_w - 5, int(sx)))
        sy_safe = max(5, min(screen_h - 5, int(sy)))
        
        pyautogui.moveTo(sx_safe, sy_safe, _pause=False)

    def click_izquierdo(self):
        pyautogui.click()
        
    def mouse_down(self):
        pyautogui.mouseDown()
        
    def mouse_up(self):
        pyautogui.mouseUp()
        
    def click_derecho(self):
        pyautogui.rightClick()
        
    def scroll_vertical(self, amount):
        pyautogui.scroll(int(amount))

    def zoom(self, delta):
        if delta != 0:
            pyautogui.keyDown('ctrl')
            import time
            time.sleep(0.02) # Tiempo para que Windows registre el modificador
            pyautogui.scroll(int(delta))
            pyautogui.keyUp('ctrl')
