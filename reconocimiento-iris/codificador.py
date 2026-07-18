"""
codificador.py — Extracción de características con filtros de Gabor (IrisCode)

El iris normalizado se divide en bloques de 16×16 píxeles. Sobre cada bloque
se aplica un filtro de Gabor complejo que extrae la fase local de la textura.
El signo de la parte real e imaginaria produce un código binario compacto
denominado IrisCode (Daugman, 1994).

Referencia teórica: Sección 2 del trabajo de titulación.
"""

import cv2 as cv
import numpy as np


def codificar_iris(img, dr=16, dtheta=16, alpha=0.4):
    """
    Genera el IrisCode binario de un iris normalizado.

    Parámetros
    ----------
    img : ndarray  shape (64, 512, 3) o (64, 512)
        Imagen de iris normalizada (salida de normalizador.py).
    dr : int
        Altura de cada bloque (divisor de 64).
    dtheta : int
        Ancho de cada bloque (divisor de 512).
    alpha : float
        Parámetro de escala del filtro Gabor (β = 1/alpha).

    Retorna
    -------
    codigo : ndarray  shape (8, 32) dtype=uint8
        IrisCode binario listo para comparación con distancia de Hamming.
    """
    # Convertir a escala de grises si viene en BGR
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Normalización estadística (media 0, varianza 1)
    img = img.astype(np.float64)
    img = (img - img.mean()) / (img.std() + 1e-8)

    # Convolución con filtro Gabor complejo
    real, imag = _convolucionar_gabor(img, w=16, alpha=alpha,
                                      beta=1/alpha, dr=dr, dtheta=dtheta)

    # Suma por bloques y binarización por signo
    n_bloques_r = img.shape[0] // dr
    n_bloques_t = img.shape[1] // dtheta

    real = real.reshape(n_bloques_r, dr, n_bloques_t, dtheta)
    imag = imag.reshape(n_bloques_r, dr, n_bloques_t, dtheta)

    suma_real = real.sum(axis=(1, 3))
    suma_imag = imag.sum(axis=(1, 3))

    codigo = np.concatenate([suma_real, suma_imag], axis=0)
    codigo = (codigo >= 0).astype(np.uint8)

    return codigo


# ---------------------------------------------------------------------------
# Filtros de Gabor
# ---------------------------------------------------------------------------

def _gabor(rho, phi, w, theta0, r0, alpha, beta):
    """
    Wavelet de Gabor bidimensional en coordenadas polares.

    La parte oscilante (e^{-iw(θ₀−φ)}) extrae la fase de la textura.
    Las envolventes gaussianas controlan la localización radial y angular.
    """
    return (
        np.exp(-w * 1j * (theta0 - phi))
        * np.exp(-((rho - r0) ** 2) / alpha ** 2)
        * np.exp(-((phi - theta0) ** 2) / beta ** 2)
    )


def _convolucionar_gabor(img, w, alpha, beta, dr, dtheta):
    """
    Aplica la wavelet de Gabor a toda la imagen mediante operaciones
    matriciales (sin bucles explícitos por píxel).

    Retorna las partes real e imaginaria de la respuesta.
    """
    # Coordenada radial ρ ∈ [0, 1] replicada para cada bloque
    rho = np.tile(np.linspace(0, 1, dr), (dtheta, 1)).T
    rho_bloques = np.tile(rho, (img.shape[0] // dr, img.shape[1] // dtheta))

    # Coordenadas angulares φ ∈ [−π, π]
    x = np.linspace(0, 1, dr)
    x_bloques = np.tile(x, img.shape[0] // dr)
    y = np.linspace(-np.pi, np.pi, dtheta)
    y_bloques = np.tile(y, img.shape[1] // dtheta)

    xx, yy = np.meshgrid(x_bloques, y_bloques)
    valor_gabor = _gabor(xx, yy, w, 0, 0.5, alpha, beta).T

    respuesta = rho_bloques * img
    return respuesta * np.real(valor_gabor), respuesta * np.imag(valor_gabor)
