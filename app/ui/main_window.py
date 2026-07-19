import customtkinter as ctk
import time
from app.camera.frame_processor import TouchWallEngine
from app.ui.settings_window import SettingsWindow
from app.config.settings_manager import load_settings

class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración básica de la ventana (Widget Estilo Móvil)
        self.title("TouchWall")
        self.geometry("320x650")
        self.minsize(320, 650)
        self.resizable(False, False)
        
        # Cargar configuración y aplicar Tema
        config = load_settings()
        tema_guardado = config.get("general", {}).get("Tema", "Oscuro")
        
        if tema_guardado == "Claro":
            ctk.set_appearance_mode("Light")
        elif tema_guardado == "Sistema":
            ctk.set_appearance_mode("System")
        else:
            ctk.set_appearance_mode("Dark")
            
        ctk.set_default_color_theme("blue")
        
        # Instanciar el Motor de TouchWall (sin iniciar aún)
        self.engine = TouchWallEngine(ui_callback=self.log_callback)
        self.is_active = False

        self._build_ui()
        
        # Loop de actualización del Dashboard
        self.update_dashboard()

    def _build_ui(self):
        # Limpiar ventana
        for widget in self.winfo_children():
            widget.destroy()
            
        self.grid_columnconfigure(0, weight=1)
        
        # ─── HEADER ───
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, pady=(30, 20), sticky="n")
        
        self.lbl_title = ctk.CTkLabel(self.header_frame, text="TOUCH WALL", font=ctk.CTkFont(size=24, weight="bold", family="Courier"))
        self.lbl_title.pack()
        
        estado_txt = "🟢 Activo" if self.is_active else "● Sistema listo"
        color_txt = "#00FF00" if self.is_active else "gray"
        self.lbl_status = ctk.CTkLabel(self.header_frame, text=estado_txt, font=ctk.CTkFont(size=14), text_color=color_txt)
        self.lbl_status.pack(pady=(5,0))
        
        # Línea separadora
        self.div1 = ctk.CTkFrame(self, height=2, fg_color="gray30")
        self.div1.grid(row=1, column=0, sticky="ew", padx=20)

        # ─── STATS ───
        self.stats_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.stats_frame.grid(row=2, column=0, pady=20, padx=40, sticky="ew")
        
        if self.is_active:
            self._build_active_stats()
        else:
            self._build_inactive_stats()

        # Línea separadora
        self.div2 = ctk.CTkFrame(self, height=2, fg_color="gray30")
        self.div2.grid(row=3, column=0, sticky="ew", padx=20)

        # ─── BOTÓN PRINCIPAL ───
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.grid(row=4, column=0, pady=30)
        
        if self.is_active:
            self.btn_main = ctk.CTkButton(self.btn_frame, text="■ Detener", width=200, height=50, fg_color="#C0392B", hover_color="#922B21", font=ctk.CTkFont(size=16, weight="bold"), command=self.detener_sistema)
            self.btn_main.pack()
        else:
            self.btn_main = ctk.CTkButton(self.btn_frame, text="▶ Iniciar Control", width=200, height=50, fg_color="#27AE60", hover_color="#1E8449", font=ctk.CTkFont(size=16, weight="bold"), command=self.iniciar_sistema)
            self.btn_main.pack()

        # Línea separadora
        self.div3 = ctk.CTkFrame(self, height=2, fg_color="gray30")
        self.div3.grid(row=5, column=0, sticky="ew", padx=20)

        # ─── MENÚ SECUNDARIO ───
        if not self.is_active:
            self.menu_frame = ctk.CTkFrame(self, fg_color="transparent")
            self.menu_frame.grid(row=6, column=0, pady=20)
            
            ctk.CTkButton(self.menu_frame, text="Calibrar Proyector", fg_color="transparent", text_color=("black", "white"), hover_color="gray30", command=self.forzar_calibracion).pack(pady=5)
            ctk.CTkButton(self.menu_frame, text="Configuración", fg_color="transparent", text_color=("black", "white"), hover_color="gray30", command=self.abrir_configuracion).pack(pady=5)
            ctk.CTkButton(self.menu_frame, text="Ayuda", fg_color="transparent", text_color=("black", "white"), hover_color="gray30").pack(pady=5)

    def abrir_configuracion(self):
        if not hasattr(self, "settings_window") or self.settings_window is None or not self.settings_window.winfo_exists():
            self.settings_window = SettingsWindow(self)
        else:
            self.settings_window.focus()

    def _build_inactive_stats(self):
        self._add_stat_row(self.stats_frame, "Cámara", "● Lista", "gray")
        
        if self.engine and self.engine.matriz_h is not None:
            self._add_stat_row(self.stats_frame, "Proyector", "● Calibrado", "#00AA00")
        else:
            self._add_stat_row(self.stats_frame, "Proyector", "● Sin Calibrar", "orange")
            
        self._add_stat_row(self.stats_frame, "Mano", "● Apagado", "gray")
        self._add_stat_row(self.stats_frame, "FPS", "--", ("black", "white"))

    def _build_active_stats(self):
        self._add_stat_row(self.stats_frame, "Tiempo activo", "00:00:00", ("black", "white"), key="tiempo")
        self._add_stat_row(self.stats_frame, "FPS", "0", ("black", "white"), key="fps")
        self._add_stat_row(self.stats_frame, "Mano", "Buscando...", "orange", key="mano")
        self._add_stat_row(self.stats_frame, "Cursor", "Activo", "#00AA00", key="cursor")

    def _add_stat_row(self, parent, title, value, color, key=None):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")
        lbl_val = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=14), text_color=color)
        lbl_val.pack(anchor="w")
        
        if key:
            if not hasattr(self, 'stat_labels'):
                self.stat_labels = {}
            self.stat_labels[key] = lbl_val

    def iniciar_sistema(self):
        self.engine.start()
        self.is_active = True
        self._build_ui()

    def detener_sistema(self):
        self.engine.stop()
        self.engine = TouchWallEngine(ui_callback=self.log_callback)
        self.is_active = False
        self._build_ui()

    def forzar_calibracion(self):
        self.iniciar_sistema()
        self.engine.iniciar_calibracion()

    def log_callback(self, msj):
        pass # Por ahora ignoramos los logs de texto largo

    def update_dashboard(self):
        """Actualiza los valores en tiempo real leyendo del Engine."""
        if self.is_active and hasattr(self, 'stat_labels'):
            stats = self.engine.get_stats()
            
            # FPS
            self.stat_labels["fps"].configure(text=str(stats["fps"]))
            
            # Mano
            if stats["mano_detectada"]:
                self.stat_labels["mano"].configure(text="Detectada", text_color="#00FF00")
            else:
                self.stat_labels["mano"].configure(text="Buscando...", text_color="orange")
                
            # Tiempo activo (formateado HH:MM:SS)
            t_seg = int(stats["tiempo_activo"])
            h = t_seg // 3600
            m = (t_seg % 3600) // 60
            s = t_seg % 60
            self.stat_labels["tiempo"].configure(text=f"{h:02d}:{m:02d}:{s:02d}")
            
            # Cursor / Estado
            if stats["modo"] == "CALIBRACION":
                self.stat_labels["cursor"].configure(text="Calibrando...", text_color="orange")
            else:
                self.stat_labels["cursor"].configure(text="Activo", text_color="#00FF00")

        self.after(500, self.update_dashboard)
