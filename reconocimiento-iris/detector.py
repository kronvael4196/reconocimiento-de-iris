"""
detector.py — Detección de pupila e iris mediante Canny + Transformada de Hough

Mejora sobre la detección por binarización simple (Poleski, 2021):
  - cv2.GaussianBlur()    suaviza la imagen antes de buscar bordes
  - cv2.Canny()           detecta bordes anatómicos (pupila / iris)
  - cv2.HoughCircles()    parametriza las circunferencias con votos

Referencia: Sección 2.5 del trabajo de titulación.
API verificada contra OpenCV 4.13.0.
"""

import cv2
import numpy as np


def detectar_pupila(img_gris):
    """
    Localiza la pupila mediante Canny + HoughCircles.

    La imagen debe estar en escala de grises y ya suavizada con GaussianBlur.
    Se prefiere el círculo más cercano al centro de la imagen (heurística
    anatómica: la pupila ocupa la zona central del ojo).

    Parámetros
    ----------
    img_gris : np.ndarray  uint8, shape (H, W)
        Imagen del ojo en escala de grises, preprocesada.

    Retorna
    -------
    centro : tuple[int, int]  — (x, y)
    radio  : int
    """
    H, W = img_gris.shape

    r_min = max(10, int(W * 0.05))
    r_max = int(W * 0.30)

    # HoughCircles espera imagen uint8 de un solo canal.
    # param1 = umbral alto de Canny interno; param2 = umbral del acumulador.
    circulos = cv2.HoughCircles(
        img_gris,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=80,
        param2=30,
        minRadius=r_min,
        maxRadius=r_max,
    )

    if circulos is not None:
        # circulos tiene forma (1, N, 3): [[[x, y, r], ...]]
        circulos = np.round(circulos[0]).astype(int)
        cx_img, cy_img = W // 2, H // 2
        mejor = min(circulos,
                    key=lambda c: (c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2)
        return (int(mejor[0]), int(mejor[1])), int(mejor[2])

    return _fallback_pupila(img_gris, r_min, r_max)


def detectar_iris(img_gris, cx, cy, r_pupila):
    """
    Estima el radio exterior del iris expandiendo desde la pupila.

    Recorre círculos concéntricos hasta detectar la transición
    iris → esclerótica (aumento brusco de luminosidad media).

    Parámetros
    ----------
    img_gris : np.ndarray  uint8, shape (H, W)
    cx, cy   : int   — centro de la pupila
    r_pupila : int   — radio de la pupila

    Retorna
    -------
    centro : tuple[int, int]  — mismo centro que la pupila
    radio  : int              — radio exterior del iris
    """
    H, W = img_gris.shape
    r_max_seguro = min(cx, cy, H - cy, W - cx) - 2
    r_base = min(round(2.0 * r_pupila), r_max_seguro)

    mascara = np.zeros(img_gris.shape, dtype=np.uint8)
    cv2.circle(mascara, (cx, cy), r_base, 1, 1)
    n_pixeles = max(int(np.sum(mascara)), 1)
    brillo_ref = float(np.sum(mascara * img_gris)) / n_pixeles
    umbral = brillo_ref * 1.04   # 4 % de aumento → entramos en esclerótica

    for r in range(r_base, r_max_seguro):
        mascara[:] = 0
        cv2.circle(mascara, (cx, cy), r, 1, 1)
        n = max(int(np.sum(mascara)), 1)
        if float(np.sum(mascara * img_gris)) / n > umbral:
            return (cx, cy), r

    return (cx, cy), min(round(3.0 * r_pupila), r_max_seguro)


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _fallback_pupila(img_gris, r_min, r_max):
    """
    Detección de reserva por binarización (método Poleski, 2021).
    Se invoca solo cuando HoughCircles no converge.
    """
    P = float(np.sum(img_gris)) / img_gris.size
    _, binaria = cv2.threshold(img_gris, P / 2, 255, cv2.THRESH_BINARY)

    kernel = np.ones((4, 4), dtype=np.uint8)
    binaria_inv = cv2.bitwise_not(binaria)
    abierta = cv2.bitwise_not(
        cv2.morphologyEx(binaria_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    )

    contornos, _ = cv2.findContours(
        cv2.bitwise_not(abierta), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        H, W = img_gris.shape
        return (W // 2, H // 2), r_min

    cnt = max(contornos, key=cv2.contourArea)
    (x, y), radio = cv2.minEnclosingCircle(cnt)
    radio = int(np.clip(round(radio), r_min, r_max))
    return (round(int(x)), round(int(y))), radio
