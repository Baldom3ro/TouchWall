import customtkinter as ctk
import threading
import copy
from app.camera.scanner import get_available_cameras
from app.config.settings_manager import load_settings, save_settings

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("TouchWall - Configuración")
        self.geometry("600x550")
        self.minsize(600, 550)
        
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.engine = parent.engine
        self.config = load_settings()
        
        # Asegurar subclaves
        for k in ["general", "camara", "gestos", "avanzado"]:
            if k not in self.config:
                self.config[k] = {}
                
        self.config_borrador = copy.deepcopy(self.config)
        
        self.sliders = {}
        self.chk_gestos = {}
        
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=(10, 5))
        
        self.tab_general = self.tabview.add("General")
        self.tab_camara = self.tabview.add("Cámara")
        self.tab_proyeccion = self.tabview.add("Proyección")
        self.tab_gestos = self.tabview.add("Gestos")
        self.tab_avanzado = self.tabview.add("Avanzado")
        self.tab_acerca = self.tabview.add("Acerca de")
        
        self._build_general()
        self._build_camara()
        self._build_proyeccion()
        self._build_gestos()
        self._build_avanzado()
        self._build_acerca()
        
        # Botón Guardar
        self.btn_guardar = ctk.CTkButton(self, text="Guardar Configuración", state="disabled", fg_color="gray30", command=self._guardar)
        self.btn_guardar.pack(pady=10)

    def _build_general(self):
        self.var_win = ctk.BooleanVar(value=self.config_borrador["general"].get("iniciar_windows", False))
        ctk.CTkCheckBox(self.tab_general, text="Iniciar con Windows", variable=self.var_win, command=self._on_draft_change).pack(anchor="w", pady=10, padx=20)
        
        self.var_min = ctk.BooleanVar(value=self.config_borrador["general"].get("abrir_minimizado", False))
        ctk.CTkCheckBox(self.tab_general, text="Abrir minimizado", variable=self.var_min, command=self._on_draft_change).pack(anchor="w", pady=10, padx=20)
        
        self.var_notif = ctk.BooleanVar(value=self.config_borrador["general"].get("notificaciones", True))
        ctk.CTkCheckBox(self.tab_general, text="Mostrar notificaciones", variable=self.var_notif, command=self._on_draft_change).pack(anchor="w", pady=10, padx=20)
        
        ctk.CTkLabel(self.tab_general, text="Tema", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(20, 5), padx=20)
        self.opt_tema = ctk.CTkOptionMenu(self.tab_general, values=["Claro", "Oscuro", "Sistema"], command=lambda v: self._on_draft_change())
        self.opt_tema.set(self.config_borrador["general"].get("Tema", "Oscuro"))
        self.opt_tema.pack(anchor="w", padx=20)

    def _build_camara(self):
        ctk.CTkLabel(self.tab_camara, text="Cámara", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(10, 5), padx=20)
        self.opt_camara = ctk.CTkOptionMenu(self.tab_camara, values=["Buscando cámaras..."], command=lambda v: self._on_draft_change())
        self.opt_camara.pack(anchor="w", padx=20)
        
        threading.Thread(target=self._escanear_camaras, daemon=True).start()
        
        info_frame = ctk.CTkFrame(self.tab_camara, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(info_frame, text="Resolución:").grid(row=0, column=0, sticky="w", pady=5)
        ctk.CTkLabel(info_frame, text="1280x720 (Optimizada)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, sticky="w", padx=20, pady=5)
        
        self._add_slider_row(self.tab_camara, "Brillo", self.config_borrador["camara"].get("Brillo", 50))
        self._add_slider_row(self.tab_camara, "Contraste", self.config_borrador["camara"].get("Contraste", 50))
        
        self.var_expo = ctk.BooleanVar(value=self.config_borrador["camara"].get("exposicion_auto", True))
        ctk.CTkCheckBox(self.tab_camara, text="Exposición automática", variable=self.var_expo, command=self._on_draft_change).pack(anchor="w", pady=20, padx=20)

    def _build_proyeccion(self):
        ctk.CTkLabel(self.tab_proyeccion, text="Configuraciones de Proyección", font=ctk.CTkFont(weight="bold", size=16)).pack(pady=20)
        ctk.CTkLabel(self.tab_proyeccion, text="(En desarrollo. Utilice el botón de calibración en la ventana principal)", text_color="gray").pack()

    def _build_acerca(self):
        ctk.CTkLabel(self.tab_acerca, text="TouchWall", font=ctk.CTkFont(weight="bold", size=24)).pack(pady=(30, 5))
        ctk.CTkLabel(self.tab_acerca, text="Versión 1.0 (En Desarrollo)").pack()

    def _build_gestos(self):
        frame = ctk.CTkScrollableFrame(self.tab_gestos, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        gestos = ["Mover cursor", "Click", "Doble click", "Scroll", "Zoom", "Arrastrar"]
        for g in gestos:
            var = ctk.BooleanVar(value=self.config_borrador["gestos"].get(g, True))
            chk = ctk.CTkCheckBox(frame, text=g, variable=var, command=self._on_draft_change)
            chk.pack(anchor="w", pady=10, padx=10)
            self.chk_gestos[g] = var

    def _build_avanzado(self):
        self._add_slider_row(self.tab_avanzado, "Sensibilidad", self.config_borrador["avanzado"].get("Sensibilidad", 50))
        self._add_slider_row(self.tab_avanzado, "Suavizado", self.config_borrador["avanzado"].get("Suavizado", 50))
        self._add_slider_row(self.tab_avanzado, "Velocidad", self.config_borrador["avanzado"].get("Velocidad", 50))

    def _add_slider_row(self, parent, label_text, default_val):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        ctk.CTkLabel(frame, text=label_text, width=100, anchor="w").pack(side="left")
        
        slider = ctk.CTkSlider(frame, from_=0, to=100, command=lambda v: self._on_draft_change())
        slider.pack(side="left", fill="x", expand=True, padx=(10, 0))
        slider.set(default_val)
        self.sliders[label_text] = slider

    def _escanear_camaras(self):
        camaras = get_available_cameras()
        nombres = [f"{idx} - {nombre}" for idx, nombre in camaras.items()]
        
        def _actualizar():
            if self.winfo_exists():
                self.opt_camara.configure(values=nombres)
                guardada = self.config_borrador["camara"].get("nombre", "")
                if guardada in nombres:
                    self.opt_camara.set(guardada)
                else:
                    self.opt_camara.set(nombres[0])
        self.after(0, _actualizar)

    def _on_draft_change(self):
        """Actualiza el borrador leyendo de la UI y verifica si hay cambios vs el original."""
        if not self.winfo_exists(): return
        
        # General
        self.config_borrador["general"]["iniciar_windows"] = self.var_win.get()
        self.config_borrador["general"]["abrir_minimizado"] = self.var_min.get()
        self.config_borrador["general"]["notificaciones"] = self.var_notif.get()
        self.config_borrador["general"]["Tema"] = self.opt_tema.get()
        
        # Cámara
        self.config_borrador["camara"]["nombre"] = self.opt_camara.get()
        if "Brillo" in self.sliders: self.config_borrador["camara"]["Brillo"] = int(self.sliders["Brillo"].get())
        if "Contraste" in self.sliders: self.config_borrador["camara"]["Contraste"] = int(self.sliders["Contraste"].get())
        self.config_borrador["camara"]["exposicion_auto"] = self.var_expo.get()
        
        # Gestos
        for g, var in self.chk_gestos.items():
            self.config_borrador["gestos"][g] = var.get()
            
        # Avanzado
        for k in ["Sensibilidad", "Suavizado", "Velocidad"]:
            if k in self.sliders:
                self.config_borrador["avanzado"][k] = int(self.sliders[k].get())
                
        # Probar en vivo en el motor (solo gestos y avanzado)
        if self.engine:
            self.engine.settings["gestos"] = self.config_borrador["gestos"]
            self.engine.settings["avanzado"] = self.config_borrador["avanzado"]
            self.engine.load_dynamic_settings()
            
        # Verificar si difiere del original
        if self.config_borrador != self.config:
            self.btn_guardar.configure(state="normal", fg_color="#27AE60")
        else:
            self.btn_guardar.configure(state="disabled", fg_color="gray30")

    def _guardar(self):
        """Aplica y guarda."""
        self.config = copy.deepcopy(self.config_borrador)
        save_settings(self.config)
        
        tema = self.config["general"]["Tema"]
        
        # IMPORTANTE: Liberar el foco modal ANTES de cambiar el tema global
        # CustomTkinter se congela si se cambia el tema mientras un Toplevel tiene grab_set()
        self.grab_release()
        
        if tema == "Claro": ctk.set_appearance_mode("Light")
        elif tema == "Oscuro": ctk.set_appearance_mode("Dark")
        else: ctk.set_appearance_mode("System")
        
        self.destroy()

    def _on_closing(self):
        # Si cerramos con la X, revertimos el motor a la config original guardada
        if self.engine:
            self.engine.settings = load_settings()
            self.engine.load_dynamic_settings()
            
        self.grab_release()
        self.destroy()
