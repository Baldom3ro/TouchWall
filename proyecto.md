# Proyecto: Proyección Interactiva (Touch Wall)

Este proyecto tiene como objetivo transformar cualquier superficie de proyección ordinaria en una pantalla táctil o interactiva utilizando una cámara y técnicas de visión por computadora.

A continuación se detallan las ideas iniciales, el roadmap secuencial de objetivos y los requisitos del sistema.

---

## 📋 Roadmap Secuencial de Objetivos

Para construir el proyecto de manera progresiva y validar cada paso antes de avanzar, seguiremos la siguiente secuencia de desarrollo:

```mermaid
graph TD
    0[0. Configurar repositorio y entorno] --> 1[1. Iniciar cámara]
    1 --> 2[2. Detectar mano]
    2 --> 3[3. Detectar dedo índice]
    3 --> 4[4. Mover un círculo rojo]
    4 --> 5[5. Mover cursor]
    5 --> 6[6. Dibujar]
    6 --> 7[7. Detectar gestos]
    7 --> 8[8. Botones]
    8 --> 9[9. Calibración]
    9 --> 10[10. Proyector]
    10 --> 11[11. Multitouch (2 Manos)]
```

### Detalle de cada objetivo:
0. **Configurar repositorio y entorno**: Inicializar Git, configurar el repositorio remoto en GitHub, y crear el entorno virtual de Python con las dependencias necesarias.
1. **Iniciar cámara**: Levantar el flujo de video en tiempo real de la webcam usando OpenCV.
2. **Detectar mano**: Implementar el modelo de MediaPipe Hands para reconocer la mano en el flujo de video.
3. **Detectar dedo índice**: Identificar el landmark específico de la punta del dedo índice (Landmark #8 en MediaPipe).
4. **Mover un círculo rojo**: Dibujar un círculo rojo en una ventana gráfica de OpenCV que siga la posición del dedo índice en tiempo real.
5. **Mover cursor**: Usar las coordenadas del dedo para mover el cursor real del sistema operativo (usando una librería como `pyautogui` o `pynput`).
6. **Dibujar**: Implementar una pizarra digital donde se pueda pintar en pantalla arrastrando el dedo índice.
7. **Detectar gestos**: Programar gestos sencillos (por ejemplo, unir el pulgar y el índice para simular "hacer click" o "comenzar a dibujar").
8. **Botones**: Crear botones virtuales en pantalla que realicen acciones cuando el dedo interactúe con su área (por ejemplo, cambiar color de dibujo, borrar lienzo).
9. **Calibración**: Implementar una matriz de homografía (usando 4 esquinas de calibración) para mapear correctamente las coordenadas del dedo de la cámara a la pantalla proyectada, corrigiendo deformaciones de perspectiva.
10. **Proyector**: Integrar todo el sistema proyectándolo sobre una superficie física (pared, mesa), realizando los ajustes finales de luz y posición.
11. **Multitouch (2 Manos)**: Ampliar la detección para trackear ambas manos simultáneamente, abriendo la puerta a gestos complejos (como el pellizco para hacer zoom) o controlar distintas áreas al mismo tiempo.

---

## ⚙️ Entorno y Recursos Mínimos (Fase Inicial)

Para la fase inicial, utilizaremos los recursos mínimos indispensables para que el desarrollo sea ágil y no requiera hardware especializado:

*   **Hardware**: 
    *   Una laptop.
    *   Una webcam (integrada o externa).
*   **Software y Lenguaje**:
    *   **Python**: Lenguaje principal de desarrollo.
    *   **OpenCV**: Para la captura, procesamiento y visualización de video (`opencv-python`).
    *   **MediaPipe**: Para la detección rápida y precisa de manos (`mediapipe`).
*   **Entorno Físico**:
    *   Un espacio bien iluminado para facilitar la detección de la webcam y evitar ruidos en el sensor de imagen.

---

## 🏗️ Arquitectura

### 4. Gestor de Configuraciones (Settings UI)
1. **Paso 1: Ajuste Fino del Tracking.** Resolver la sensibilidad extrema del cursor y facilitar la activación de clics ajustando los parámetros matemáticos y de estabilización.
2. **Paso 2: Menú de Configuraciones UI.** Crear las pestañas de configuración usando `CustomTkinter` (General, Cámara, Proyección, Gestos, Avanzado, Acerca de).
3. **Paso 3: Selector Dinámico de Cámaras.** Implementar código OpenCV para detectar cámaras disponibles y mostrar un menú desplegable en la interfaz.
4. **Paso 4: Controles de Gestos.** Añadir *checkboxes* para activar/desactivar individualmente: Mover cursor, Click, Doble Click, Scroll, Zoom y Arrastrar.
5. **Paso 5: Sliders de Sensibilidad.** Añadir barras ajustables (Sensibilidad, Suavizado, Velocidad) que modifiquen en tiempo real el comportamiento de la IA.
6. **Paso 6: Persistencia de Ajustes.** Guardar todas estas opciones en un archivo `config.json` para que el sistema las recuerde al reiniciar.

### 5. Empaquetado y Distribución
- Compilar el proyecto entero (incluyendo el modelo `hand_landmarker.task` y la interfaz gráfica) usando `PyInstaller` para crear un ejecutable `.exe` standalone. módulos especializados:

```text
TouchWall/
├── main.py                     # Punto de entrada
├── requirements.txt
├── README.md
├── LICENSE
│
├── app/
│   ├── __init__.py
│   ├── ui/                     # Toda la interfaz (CustomTkinter)
│   │   ├── main_window.py
│   │   ├── calibration.py
│   │   ├── settings.py
│   │   ├── dialogs.py
│   │   └── widgets.py
│   ├── camera/                 # Captura de video
│   │   ├── camera_manager.py
│   │   ├── frame_processor.py
│   │   └── calibration.py
│   ├── tracking/               # Inteligencia artificial
│   │   ├── hand_detector.py
│   │   ├── finger_tracker.py
│   │   ├── smoothing.py
│   │   └── filters.py
│   ├── gestures/               # Lógica de gestos
│   │   ├── click.py
│   │   ├── drag.py
│   │   ├── scroll.py
│   │   ├── zoom.py
│   │   └── gesture_manager.py
│   ├── projection/             # Homografía y mapeo
│   │   ├── homography.py
│   │   ├── coordinate_mapper.py
│   │   ├── projector.py
│   │   └── calibration.py
│   ├── system/                 # Control del Sistema Operativo
│   │   ├── mouse_controller.py
│   │   ├── keyboard_controller.py
│   │   ├── permissions.py
│   │   └── monitor.py
│   ├── config/                 # Configuraciones persistentes
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── default_config.json
│   ├── resources/              # Archivos multimedia
│   │   ├── icons/
│   │   ├── images/
│   │   ├── fonts/
│   │   └── sounds/
│   ├── models/                 # Modelos de Machine Learning
│   │   └── hand_landmarker.task
│   ├── utils/                  # Herramientas
│   │   ├── logger.py
│   │   ├── helpers.py
│   │   └── math_utils.py
│   └── data/                   # Archivos generados (como la matriz H)
│       └── calibration.json
│
├── installer/                  # Recursos para InnoSetup o NSIS
│   ├── setup.iss
│   └── icon.ico
├── build/                      # Creado por PyInstaller (ignorado)
├── dist/                       # Creado por PyInstaller (ignorado)
└── tests/                      # Tests unitarios
```

---

## 🛠️ Registro de Optimizaciones y Solución de Errores (Fase 4)
Durante la implementación de la Interfaz Gráfica y el Motor, se detectaron y resolvieron los siguientes retos técnicos:

1. **Caída de Rendimiento (10 FPS limit):**
   - **Problema:** OpenCV, por defecto, solicitaba video crudo (YUY2) a la cámara, lo que saturaba el ancho de banda del bus USB, limitando físicamente los FPS a 10.
   - **Solución:** Se forzó a OpenCV a solicitar compresión por hardware (`MJPG`) usando `cv2.CAP_PROP_FOURCC`. Adicionalmente, se estableció la resolución de captura en **1280x720 (720p)**, y se redujo el tamaño del frame a 640x360 milisegundos antes de inyectarlo en MediaPipe. Esto alivió la carga del procesador y restableció el rendimiento a +30 FPS sin perder precisión.
2. **Muro Invisible en PyAutoGUI:**
   - **Problema:** El cursor físico del sistema chocaba con una pared invisible al llegar al 75% de la pantalla (1280x720 píxeles).
   - **Solución:** Se eliminó la dependencia de la resolución estática de la cámara (`ANCHO`, `ALTO`) en `mouse_controller.py` y se implementó `pyautogui.size()` para leer dinámicamente los bordes reales del monitor de Windows, permitiendo usar el 100% de la superficie física.
3. **Desajuste de Homografía (Ventana de Calibración 3/4):**
   - **Problema:** Al capturar en 720p, la ventana negra de calibración se encogía, por lo que el usuario calibraba sus dedos sobre un cuadro interno, corrompiendo las matemáticas de la matriz de perspectiva.
   - **Solución:** Se forzó la ampliación por software (`cv2.resize`) del lienzo de calibración usando `pyautogui.size()` justo antes de activar el Fullscreen, garantizando que los puntos rojos siempre coincidan con las esquinas del monitor físico.
4. **Pausa Errónea (Puño Falso) al Apuntar:**
   - **Problema:** Apuntar con el dedo índice (cerrando los otros 3 dedos) activaba la regla antigua de "3 dedos cerrados = Pausa".
   - **Solución:** Se actualizó `gesture_manager.py` para exigir que los **4 dedos completos** estén cerrados para registrar el `GESTO_PUÑO`.
5. **Gesto de Doble Clic:**
   - **Solución:** Se integró un nuevo gesto híbrido (`d_double`) que mide la distancia de "pinch" entre el Pulgar y el dedo Medio, inyectando un doble clic ultrarrápido al SO.
6. **Deadlock (Congelamiento) al cambiar Temas en Tkinter:**
   - **Problema:** Cambiar de Modo Claro a Oscuro desde una ventana modal (`grab_set`) colapsaba el hilo de eventos de Tcl/Tk, congelando toda la interfaz gráfica y volviendo "invisibles" los botones.
   - **Solución:** Se corrigió el color forzado de los botones (`text_color=("black", "white")`) y se programó la ventana de configuraciones para liberar los bloqueos modales ANTES de inyectar el nuevo tema al motor visual de CustomTkinter.
7. **Arquitectura de Borradores (Drafts) en Configuraciones:**
   - **Problema:** Los gestos y ajustes visuales se sobreescribían, la previsualización en vivo no guardaba estado, y la cámara no respetaba el menú desplegable.
   - **Solución:** Se implementó un sistema inteligente de borradores (`config_borrador`). Los cambios ahora afectan al motor visual en tiempo real, pero si el usuario cierra la ventana, los cambios se descartan. Solo al presionar "Guardar Configuración" se realiza la escritura al archivo `config.json` y se recarga de forma limpia en el núcleo del sistema.
