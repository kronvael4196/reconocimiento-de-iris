"""
comparador.py — Comparación de IrisCodes mediante Distancia de Hamming Normalizada

La distancia de Hamming mide la fracción de bits que difieren entre dos códigos.
Para iris genuinos se esperan distancias < 0.32; para impostores > 0.45.

Se implementa corrección por rotación: se prueban desplazamientos columnares
(±rotacion_max) para compensar diferencias de orientación entre capturas.

Referencia: Daugman, J. (2004). How iris recognition works. IEEE TCSVT, 14(1).
"""

import numpy as np


# Umbral de decisión para autenticación
# (calibrado en experimentos con CASIA; ajustar con datos propios del instituto)
UMBRAL_AUTENTICACION = 0.38


def comparar_codigos(codigo_a, codigo_b, rotacion_max=8):
    """
    Calcula la Distancia de Hamming Normalizada entre dos IrisCodes.

    Parámetros
    ----------
    codigo_a, codigo_b : ndarray
        Códigos binarios de dos iris (misma forma).
    rotacion_max : int
        Número máximo de desplazamientos columnares a probar en cada dirección.
        Un valor de 8 compensa hasta ≈ 5.6° de rotación ocular.

    Retorna
    -------
    distancia : float
        Valor en [0, 1]. 0 = idénticos, 0.5 = aleatorios, 1 = opuestos.
    """
    distancias = []
    for desplazamiento in range(-rotacion_max, rotacion_max + 1):
        codigo_rotado = np.roll(codigo_b, desplazamiento, axis=1)
        d = np.count_nonzero(codigo_a != codigo_rotado) / codigo_a.size
        distancias.append(d)
    return float(np.min(distancias))


def es_coincidencia(distancia):
    """
    Decide si dos iris pertenecen a la misma persona.

    Parámetros
    ----------
    distancia : float
        Distancia de Hamming normalizada.

    Retorna
    -------
    bool
        True si la distancia está por debajo del umbral de autenticación.
    """
    return distancia < UMBRAL_AUTENTICACION


def identificar_usuario(codigo_consulta, base_codigos, rotacion_max=8):
    """
    Busca al usuario más parecido en la base de datos.

    Parámetros
    ----------
    codigo_consulta : ndarray
        IrisCode del iris a identificar.
    base_codigos : dict[str, ndarray]
        Diccionario {id_usuario: codigo} con todos los registros.
    rotacion_max : int
        Parámetros de corrección por rotación.

    Retorna
    -------
    id_usuario : str o None
        ID del usuario identificado, o None si no supera el umbral.
    distancia_min : float
        Distancia al código más cercano encontrado.
    """
    if not base_codigos:
        return None, 1.0

    resultados = [
        (comparar_codigos(codigo_consulta, codigo, rotacion_max), uid)
        for uid, codigo in base_codigos.items()
    ]
    distancia_min, id_mejor = min(resultados)

    if es_coincidencia(distancia_min):
        return id_mejor, distancia_min
    return None, distancia_min
