import time

class Estabilizador:
    """
    Filtro One-Euro Simplificado para estabilización del cursor.
    """
    def __init__(self, alpha_min=0.05, alpha_max=0.8, umbral_velocidad=0.05):
        self.x_prev = None
        self.y_prev = None
        self.t_prev = None
        
        # Parámetros ajustables
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.umbral_velocidad = umbral_velocidad

    def actualizar(self, x, y):
        t_actual = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.y_prev = y
            self.t_prev = t_actual
            return x, y
            
        dt = t_actual - self.t_prev
        if dt == 0: dt = 0.001
        
        # Calcular velocidad (distancia / tiempo)
        vx = (x - self.x_prev) / dt
        vy = (y - self.y_prev) / dt
        v = (vx**2 + vy**2)**0.5
        
        # Factor de mezcla dinámico (LERP)
        # Si se mueve rápido, confía en el valor nuevo (menor latencia)
        # Si se mueve lento, confía en el valor anterior (mayor suavidad)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * min(1.0, v / self.umbral_velocidad)
        
        # LERP (Linear Interpolation)
        x_suave = self.x_prev + alpha * (x - self.x_prev)
        y_suave = self.y_prev + alpha * (y - self.y_prev)
        
        self.x_prev = x_suave
        self.y_prev = y_suave
        self.t_prev = t_actual
        
        return x_suave, y_suave
