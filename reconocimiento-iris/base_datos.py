"""
base_datos.py — Gestión de usuarios y plantillas de iris para el sistema de biblioteca

Almacena los IrisCodes en disco como archivos .npy (NumPy) y los metadatos
de los usuarios en un JSON. No requiere ningún motor de base de datos externo.

Estructura en disco:
    data/
      usuarios/
        usuarios.json          ← metadatos (nombre, carnet, tipo, fecha_registro)
        <id_usuario>.npy       ← IrisCode binario serializado
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

DIRECTORIO_DATOS = Path(__file__).parent.parent / "data" / "usuarios"
ARCHIVO_META     = DIRECTORIO_DATOS / "usuarios.json"

def inicializar():
    """Crea la estructura de directorios y el archivo de metadatos si no existen."""
    DIRECTORIO_DATOS.mkdir(parents=True, exist_ok=True)
    if not ARCHIVO_META.exists():
        ARCHIVO_META.write_text(json.dumps({}, indent=2, ensure_ascii=False))

def registrar_usuario(id_usuario, nombre, carnet, tipo, codigo_iris):
    """
    Registra un nuevo usuario en la base de datos.

    Parámetros
    ----------
    id_usuario : str
        Identificador único (ej. número de cédula).
    nombre : str
        Nombre completo del usuario.
    carnet : str
        Número de carnet institucional.
    tipo : str
        'estudiante', 'docente' o 'administrativo'.
    codigo_iris : ndarray
        IrisCode generado por codificador.py.
    """
    inicializar()

    # Guardar código de iris como archivo binario NumPy
    ruta_codigo = DIRECTORIO_DATOS / f"{id_usuario}.npy"
    np.save(str(ruta_codigo), codigo_iris)

    # Actualizar metadatos JSON
    meta = _leer_meta()
    meta[id_usuario] = {
        "nombre": nombre,
        "carnet": carnet,
        "tipo": tipo,
        "fecha_registro": datetime.now().isoformat(timespec="seconds"),
    }
    _escribir_meta(meta)
    print(f"✓ Usuario '{nombre}' registrado con ID: {id_usuario}")


def cargar_todos_los_codigos():
    """
    Carga todos los IrisCodes registrados desde disco.

    Retorna
    -------
    dict[str, ndarray]
        Diccionario {id_usuario: codigo_iris}.
    """
    inicializar()
    codigos = {}
    for archivo in DIRECTORIO_DATOS.glob("*.npy"):
        uid = archivo.stem
        codigos[uid] = np.load(str(archivo))
    return codigos

def obtener_info_usuario(id_usuario):
    """
    Recupera los metadatos de un usuario por su ID.

    Retorna
    -------
    dict o None
    """
    meta = _leer_meta()
    return meta.get(id_usuario)


def listar_usuarios():
    """Imprime un resumen de todos los usuarios registrados."""
    meta = _leer_meta()
    if not meta:
        print("No hay usuarios registrados.")
        return
    print(f"\n{'ID':<15} {'Nombre':<30} {'Carnet':<15} {'Tipo':<15} Registro")
    print("-" * 85)
    for uid, info in meta.items():
        print(f"{uid:<15} {info['nombre']:<30} {info['carnet']:<15} "
              f"{info['tipo']:<15} {info['fecha_registro']}")


def eliminar_usuario(id_usuario):
    """Elimina un usuario y su IrisCode de la base de datos."""
    meta = _leer_meta()
    if id_usuario not in meta:
        print(f"Usuario '{id_usuario}' no encontrado.")
        return
    ruta_codigo = DIRECTORIO_DATOS / f"{id_usuario}.npy"
    if ruta_codigo.exists():
        ruta_codigo.unlink()
    del meta[id_usuario]
    _escribir_meta(meta)
    print(f"Usuario '{id_usuario}' eliminado.")

# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _leer_meta():
    inicializar()
    return json.loads(ARCHIVO_META.read_text(encoding="utf-8"))

def _escribir_meta(meta):
    ARCHIVO_META.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
