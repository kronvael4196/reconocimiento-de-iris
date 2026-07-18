"""
main.py — Punto de entrada del Sistema de Reconocimiento de Iris para Biblioteca

Uso:
    python main.py registrar   → registrar nuevo usuario
    python main.py autenticar  → autenticar usuario existente
    python main.py listar      → listar usuarios registrados
    python main.py eliminar    → eliminar un usuario

Instituto Superior Tecnológico Alquimia
Carrera de Ciberseguridad — Trabajo de Titulación 2026
Autor: Vicente Alejandro Venegas Riera
"""

import sys
from base_datos import listar_usuarios, eliminar_usuario
from captura    import registrar, autenticar


def menu_registrar():
    print("\n--- REGISTRO DE NUEVO USUARIO ---")
    id_usuario = input("Número de cédula (ID único): ").strip()
    nombre     = input("Nombre completo:             ").strip()
    carnet     = input("Número de carnet:            ").strip()
    print("Tipo de usuario:")
    print("  1. Estudiante")
    print("  2. Docente")
    print("  3. Administrativo")
    opcion = input("Selecciona (1-3): ").strip()
    tipos  = {"1": "estudiante", "2": "docente", "3": "administrativo"}
    tipo   = tipos.get(opcion, "estudiante")

    registrar(id_usuario, nombre, carnet, tipo)


def menu_autenticar():
    resultado = autenticar()
    if resultado:
        print(f"\n→ Sesión iniciada para usuario: {resultado}")
    else:
        print("\n→ Autenticación fallida.")


def menu_eliminar():
    print("\n--- ELIMINAR USUARIO ---")
    listar_usuarios()
    id_usuario = input("\nID del usuario a eliminar: ").strip()
    confirmar  = input(f"¿Confirmar eliminación de '{id_usuario}'? (s/N): ").strip()
    if confirmar.lower() == 's':
        eliminar_usuario(id_usuario)


COMANDOS = {
    "registrar":  menu_registrar,
    "autenticar": menu_autenticar,
    "listar":     listar_usuarios,
    "eliminar":   menu_eliminar,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMANDOS:
        print(__doc__)
        print("Comandos disponibles:", ", ".join(COMANDOS))
        sys.exit(1)

    COMANDOS[sys.argv[1]]()


if __name__ == "__main__":
    main()
