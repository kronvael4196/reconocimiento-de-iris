"""
normalizador.py — Preprocesamiento y normalización Rubber Sheet del iris

Pipeline:
  1. cv2.cvtColor(BGR→GRAY)       reduce dimensionalidad
  2. cv2.GaussianBlur(5×5, σ=1)   atenúa ruido de alta frecuencia
  3. detectar_pupila / detectar_iris  localiza las dos circunferencias
  4. cv2.warpPolar()               desenrolla el anillo a rectángulo 512×64

Referencia: Secciones 2.1, 2.3 y 2.4 del trabajo de titulación.
API verificada contra OpenCV 4.13.0.
"""

import cv2
import numpy as np

from detector import detectar_pupila, detectar_iris


ANCHO_NORM = 512   # eje angular (θ)
ALTO_NORM  = 64    # eje radial  (ρ)


def normalizar_imagen(img):
    """
    Transforma una imagen BGR del ojo en la representación rectangular
    normalizada del iris (512 × 64 px, BGR).

    Parámetros
    ----------
    img : np.ndarray  uint8, shape (H, W, 3)
        Frame BGR capturado con la cámara.

    Retorna
    -------
    iris_norm : np.ndarray  uint8, shape (64, 512, 3)

    Levanta
    -------
    ValueError  si la imagen está vacía.
    """
    if img is None or img.size == 0:
        raise ValueError("La imagen proporcionada está vacía o es None.")

    # 1. Escala de grises
    # cv2.cvtColor: src BGR uint8 → dst GRAY uint8
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Desenfoque gaussiano
    # cv2.GaussianBlur(src, ksize, sigmaX) → dst
    # ksize debe ser tupla de enteros impares; sigmaX=1 es estándar para iris.
    gris_suave = cv2.GaussianBlur(gris, (5, 5), sigmaX=1)

    # 3. Detección geométrica
    (cx, cy), r_pupila = detectar_pupila(gris_suave)
    (_, _),   r_iris   = detectar_iris(gris_suave, cx, cy, r_pupila)

    # 4. Desenrollado polar (Rubber Sheet)
    iris_recortado = _recortar_iris(img, cx, cy, r_iris)
    return _desenrollar_iris(iris_recortado, r_iris, r_pupila)


def dibujar_deteccion(img):
    """
    Superpone los círculos de pupila e iris para depuración visual.

    Retorna
    -------
    img_anotada : np.ndarray  (copia de img con círculos superpuestos)
    cx, cy, r_pupila, r_iris : int
    """
    gris       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris_suave = cv2.GaussianBlur(gris, (5, 5), sigmaX=1)

    (cx, cy), r_pupila = detectar_pupila(gris_suave)
    (_, _),   r_iris   = detectar_iris(gris_suave, cx, cy, r_pupila)

    anotada = img.copy()
    # cv2.circle(img, center, radius, color_BGR, thickness)
    cv2.circle(anotada, (cx, cy), r_pupila, (0, 0, 255),  2)   # rojo
    cv2.circle(anotada, (cx, cy), r_iris,   (0, 255, 0),  2)   # verde
    cv2.circle(anotada, (cx, cy), 3,        (255, 0, 0), -1)   # azul

    return anotada, cx, cy, r_pupila, r_iris


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _recortar_iris(img, cx, cy, r_iris):
    """Extrae el cuadrado que rodea al iris con clip de bordes."""
    H, W = img.shape[:2]
    x1 = max(cy - r_iris, 0)
    x2 = min(cy + r_iris, H)
    y1 = max(cx - r_iris, 0)
    y2 = min(cx + r_iris, W)
    return img[x1:x2, y1:y2]


def _desenrollar_iris(img_crop, r_iris, r_pupila,
                      M=ALTO_NORM, N=ANCHO_NORM):
    """
    Aplica cv2.warpPolar para convertir el anillo del iris en un
    rectángulo de dimensiones fijas (equivalente al modelo Rubber Sheet).

    cv2.warpPolar(src, dsize, center, maxRadius, flags) → dst
      dsize     = (ancho_destino, alto_destino)  como tupla de int
      center    = (cx, cy)                       en float o int
      maxRadius = radio exterior del iris
      flags     = cv2.WARP_POLAR_LINEAR           interpolación lineal
    """
    if img_crop.size == 0:
        return np.zeros((M, N, 3), dtype=np.uint8)

    factor_int = r_pupila / max(r_iris, 1)
    alto_total = round(M / max(1.0 - factor_int, 0.01))

    H_c, W_c = img_crop.shape[:2]
    centro = (W_c // 2, H_c // 2)

    # warpPolar devuelve shape (alto_total, N, C)
    desenrollado = cv2.warpPolar(
        img_crop,
        (N, alto_total),          # dsize = (ancho, alto)
        centro,
        float(r_iris),
        cv2.WARP_POLAR_LINEAR,
    )

    # Rotar 90° para que el eje angular quede horizontal
    # cv2.rotate(src, rotateCode) → dst
    recto = cv2.rotate(desenrollado, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return recto[:M, :]           # recortar la banda del iris (sin pupila)
