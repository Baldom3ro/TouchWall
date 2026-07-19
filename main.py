import customtkinter as ctk
from app.ui.main_window import MainWindow
import sys

def main():
    # Configuración global de la apariencia de CustomTkinter
    ctk.set_appearance_mode("Dark")  # Modos: "System" (estándar), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Temas: "blue" (estándar), "green", "dark-blue"

    # Iniciar la aplicación
    app = MainWindow()
    
    # Manejar el cierre de la ventana de forma limpia
    def on_closing():
        app.destroy()
        sys.exit(0)
        
    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()

if __name__ == "__main__":
    main()
