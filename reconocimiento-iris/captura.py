"""
captura.py — Captura con webcam para registro y autenticación en la biblioteca

Flujos de trabajo:
  registrar()   → captura N muestras, promedia los IrisCodes y los guarda
  autenticar()  → captura 1 muestra y la compara contra la base de datos

Teclas durante la captura:
  ESPACIO  → capturar fotograma
  D        → activar/desactivar vista de detección (círculos Canny/Hough)
  Q / ESC  → cancelar

API de OpenCV verificada contra 4.13.0.
"""

import cv2
import numpy as np

from normalizador import normalizar_imagen, dibujar_deteccion
from codificador  import codificar_iris
from comparador   import identificar_usuario
from base_datos   import (registrar_usuario, cargar_todos_los_codigos,
                           obtener_info_usuario)

N_MUESTRAS_REGISTRO = 5

def registrar(id_usuario, nombre, carnet, tipo, camara=0):
    """
    Registra un nuevo usuario capturando N_MUESTRAS_REGISTRO imágenes del ojo.

    El IrisCode final es el promedio rebinarizado de todas las muestras,
    lo que mejora la robustez frente a variaciones de iluminación o posición.

    Parámetros
    ----------
    id_usuario : str   — número de cédula u otro ID único
    nombre     : str
    carnet     : str
    tipo       : str   — 'estudiante' | 'docente' | 'administrativo'
    camara     : int   — índice para cv2.VideoCapture (0 = predeterminado)
    """
    cap = _abrir_camara(camara)
    codigos = []
    mostrar_det = False

    print(f"\n=== REGISTRO: {nombre} ===")
    print(f"Se necesitan {N_MUESTRAS_REGISTRO} capturas. ESPACIO=capturar  D=deteccion  Q=cancelar")

    while len(codigos) < N_MUESTRAS_REGISTRO:
        ret, frame = cap.read()
        if not ret:
            print("Error al leer la cámara.")
            break

        vista = _vista_deteccion(frame) if mostrar_det else frame.copy()
        _hud(vista,
             f"Registro [{len(codigos)}/{N_MUESTRAS_REGISTRO}]",
             "ESPACIO=capturar  D=deteccion  Q=cancelar")

        # cv2.imshow(winname, mat) — muestra la imagen en una ventana
        cv2.imshow("Registro de Iris", vista)

        # cv2.waitKey(delay_ms) — espera delay ms y retorna código de tecla
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):
            codigo = _procesar(frame)
            if codigo is not None:
                codigos.append(codigo)
                print(f"  Muestra {len(codigos)}/{N_MUESTRAS_REGISTRO} ✓")
                _flash("Registro de Iris", frame)
            else:
                print("Iris no detectado — intenta de nuevo")

        elif tecla in (ord('d'), ord('D')):
            mostrar_det = not mostrar_det

        elif tecla in (ord('q'), ord('Q'), 27):
            print("Registro cancelado.")
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(codigos) == N_MUESTRAS_REGISTRO:
        plantilla = (np.mean(codigos, axis=0) >= 0.5).astype(np.uint8)
        registrar_usuario(id_usuario, nombre, carnet, tipo, plantilla)
    else:
        print("Registro incompleto.")

def autenticar(camara=0):
    """
    Autentica al usuario que coloca su ojo frente a la cámara.

    Retorna
    -------
    id_usuario : str | None  — ID del usuario autenticado, o None si falla.
    """
    base = cargar_todos_los_codigos()
    if not base:
        print("No hay usuarios registrados.")
        return None

    cap = _abrir_camara(camara)
    mostrar_det = False
    resultado = None

    print("\n=== AUTENTICACIÓN ===  ESPACIO=verificar  D=deteccion  Q=cancelar")

    while resultado is None:
        ret, frame = cap.read()
        if not ret:
            break

        vista = _vista_deteccion(frame) if mostrar_det else frame.copy()
        _hud(vista, "Autenticación", "ESPACIO=verificar  D=deteccion  Q=cancelar")
        cv2.imshow("Autenticación de Iris", vista)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord(' '):
            codigo = _procesar(frame)
            if codigo is not None:
                uid, dist = identificar_usuario(codigo, base)
                if uid:
                    info = obtener_info_usuario(uid)
                    nombre = info["nombre"] if info else uid
                    print(f"\n✓ ACCESO CONCEDIDO — {nombre}  (HD={dist:.3f})")
                    _resultado_visual("Autenticación de Iris", frame,
                                      f"ACCESO CONCEDIDO: {nombre}", (0, 180, 0))
                    resultado = uid
                else:
                    print(f"\n✗ ACCESO DENEGADO  (HD_min={dist:.3f})")
                    _resultado_visual("Autenticación de Iris", frame,
                                      "ACCESO DENEGADO", (0, 0, 200))
                    resultado = False
            else:
                print("Iris no detectado — intenta de nuevo")

        elif tecla in (ord('d'), ord('D')):
            mostrar_det = not mostrar_det

        elif tecla in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return resultado if resultado else None


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _abrir_camara(indice):
    """
    cv2.VideoCapture(index) — abre el dispositivo de captura.
    Configura resolución HD para mejorar la detección del iris.
    """
    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara {indice}. "
            "Verifica que esté conectada y no esté en uso por otra aplicación."
        )
    # cv2.VideoCapture.set(propId, value)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    return cap

def _procesar(frame):
    """Normaliza y codifica un fotograma. Retorna None si falla la detección."""
    try:
        return codificar_iris(normalizar_imagen(frame))
    except Exception:
        return None

def _vista_deteccion(frame):
    """Intenta mostrar los círculos Canny/Hough sobre el frame."""
    try:
        vista, *_ = dibujar_deteccion(frame)
        return vista
    except Exception:
        return frame.copy()

def _hud(img, titulo, instrucciones):
    """
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
    Superpone texto de instrucciones sobre el frame.
    """
    cv2.putText(img, titulo,        (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, instrucciones, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def _flash(winname, frame):
    """Parpadeo blanco como retroalimentación visual al capturar."""
    blanco = np.full_like(frame, 255)
    cv2.imshow(winname, blanco)
    cv2.waitKey(80)

def _resultado_visual(winname, frame, mensaje, color_bgr, dur_ms=2500):
    """Muestra el resultado de autenticación sobre el frame durante dur_ms."""
    img = frame.copy()
    H, W = img.shape[:2]
    # cv2.rectangle(img, pt1, pt2, color, thickness=-1 rellena)
    cv2.rectangle(img, (0, H // 2 - 45), (W, H // 2 + 45), color_bgr, -1)
    cv2.putText(img, mensaje,
                (max(W // 2 - len(mensaje) * 7, 10), H // 2 + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow(winname, img)
    cv2.waitKey(dur_ms)
