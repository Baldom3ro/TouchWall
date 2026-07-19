import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "config.json")

DEFAULT_CONFIG = {
    "gestos": {
        "Mover cursor": True,
        "Click": True,
        "Doble click": True,
        "Scroll": True,
        "Zoom": False,
        "Arrastrar": True
    },
    "avanzado": {
        "Sensibilidad": 50,
        "Suavizado": 50,
        "Velocidad": 50
    }
}

def load_settings():
    if not os.path.exists(os.path.dirname(CONFIG_FILE)):
        os.makedirs(os.path.dirname(CONFIG_FILE))
        
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                
                # Merge con defaults por si faltan claves nuevas
                for category in DEFAULT_CONFIG:
                    if category not in data:
                        data[category] = DEFAULT_CONFIG[category]
                    else:
                        for key in DEFAULT_CONFIG[category]:
                            if key not in data[category]:
                                data[category][key] = DEFAULT_CONFIG[category][key]
                return data
        except:
            return DEFAULT_CONFIG.copy()
    else:
        return DEFAULT_CONFIG.copy()

def save_settings(data):
    if not os.path.exists(os.path.dirname(CONFIG_FILE)):
        os.makedirs(os.path.dirname(CONFIG_FILE))
        
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"[Error] No se pudo guardar config.json: {e}")
