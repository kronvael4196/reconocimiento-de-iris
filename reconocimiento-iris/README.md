# Sistema de Reconocimiento de Iris para Biblioteca

**Trabajo de Titulación — Instituto Superior Tecnológico Alquimia**  
Carrera de Ciberseguridad · 2026  
Autor: Vicente Alejandro Venegas Riera  
Tutor: Ing. Fernando Albrito

---

## Descripción

Sistema biométrico de autenticación de usuarios basado en reconocimiento de iris,
implementado en Python con OpenCV. Permite registrar y autenticar a los usuarios
de la biblioteca institucional sin contacto físico.

## Pipeline del algoritmo

Imagen de ojo (webcam)
        │
        ▼

  1. Preprocesamiento        → Conversión a escala de grises + Filtro Gaussiano
        │
        ▼
  2. Detección (Canny+Hough) → Localiza pupila e iris como circunferencias
        │
        ▼
  3. Normalización (Rubber Sheet) → Desenrolla el anillo del iris a 512×64 px
        │
        ▼
  4. Codificación (Gabor)    → Genera IrisCode binario (8×32 bits)
        │
        ▼
  5. Comparación (Hamming)   → Distancia normalizada contra base de datos
        │
        ▼
  ACCESO CONCEDIDO / DENEGADO

## Estructura del proyecto

reconocimiento-de-iris/
├── src/
│   ├── main.py          ← Punto de entrada (CLI)
│   ├── captura.py       ← Captura con webcam (registro y autenticación)
│   ├── detector.py      ← Detección de pupila e iris (Canny + Hough)
│   ├── normalizador.py  ← Preprocesamiento y modelo Rubber Sheet
│   ├── codificador.py   ← Extracción de características con filtros Gabor
│   ├── comparador.py    ← Distancia de Hamming y decisión de acceso
│   └── base_datos.py    ← Persistencia de usuarios e IrisCodes
├── data/
│   └── usuarios/        ← Metadatos JSON + plantillas .npy
├── capturas/            ← Imágenes capturadas (para depuración)
├── requirements.txt
└── README.md

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/kronvael4196/reconocimiento-de-iris.git
cd reconocimiento-de-iris

# 2. Crear entorno virtual (requerido en Arch Linux / Omarchy por PEP 668)
python -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
cd src

# Registrar un nuevo usuario
python main.py registrar

# Autenticar (verificar identidad)
python main.py autenticar

# Listar usuarios registrados
python main.py listar

# Eliminar un usuario
python main.py eliminar
```

### Teclas durante la captura

| Tecla          | Acción                                                       |
|----------------|                                                              |
| `ESPACIO`      | Capturar fotograma                                           |
| `D`            | Activar/desactivar vista de detección (círculos Canny/Hough) |
| `Q` / `ESC`    | Cancelar                                                     |

## Fundamento teórico

- **Detección:** Bordes de Canny + Transformada Circular de Hough (Wildes, 1997)
- **Normalización:** Modelo Rubber Sheet (Daugman, 1994)
- **Codificación:** Filtros de Gabor complejos → IrisCode binario
- **Comparación:** Distancia de Hamming Normalizada con corrección rotacional
- **Umbral:** HD < 0.38 → mismo usuario (calibrar con datos del instituto)

## Créditos

Algoritmo base adaptado de:
> Poleski, M. (2021). *iris-recognition*. GitHub.
> <https://github.com/mateuszpoleski/iris-recognition>

Mejoras implementadas en este trabajo:

- Sustitución de binarización simple por **Canny + Hough** para detección más robusta
- Módulo de **captura en tiempo real** con webcam
- Sistema de **base de datos local** para el entorno institucional
- Documentación completa en español

## Marco legal

El tratamiento de datos biométricos se realiza en cumplimiento de la
**Ley Orgánica de Protección de Datos Personales del Ecuador (LOPDP, 2021)**.
Todo usuario debe otorgar consentimiento libre, específico e informado
antes de registrar su iris en el sistema.
