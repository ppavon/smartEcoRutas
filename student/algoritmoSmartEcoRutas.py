from __future__ import annotations

from typing import List


def solve(problem, time_limit_s: float, seed: int | None = None) -> List[List[str]]:
    """
    Plantilla para alumnos (SmartEcoRutas).

    Esta es la función que `run.py` llamará automáticamente:
        solve(problem, time_limit_s=<segundos>, seed=<entero o None>)

    Parámetros de entrada:
      - problem: instancia con métodos útiles (base_uid, dump_uid, containers_uids, time_uid, etc.).
      - time_limit_s: tiempo máximo de ejecución permitido para tu algoritmo.
      - seed: semilla opcional para reproducibilidad.

    Salida esperada:
      - list[list[str]] con rutas cerradas que terminen en ... DUMP, BASE.
      - Ejemplo de ruta: ["BASE", "c_001", "c_002", "DUMP", "BASE"].

    Recomendación:
      - Revisa heurísticos clásicos de VRP y variantes (CVRP, VRPTW, etc.).
      - Entiende bien restricciones de capacidad, tiempo y visitas a DUMP.
      - Mira `algoritmoSmartEcoRutas_simple_example.py`: es un ejemplo MUY sencillo
        pensado para entender la estructura, no para obtener rutas de alta calidad.

    Entrega del código (importante para ejecución automática):
      - Preferimos que toda la solución esté en este único archivo:
          `algoritmoSmartEcoRutas.py`
      - Si necesitas más módulos `.py`, puedes usarlos, pero todos deben estar
        en este mismo directorio `student/`.
      - Si necesitas librerías extra de Python, añade un `requirements.txt`
        también en este directorio `student/`.
      - El sistema automático instalará dependencias con:
          `pip install -r student/requirements.txt`
      - Por favor, usa el formato estándar de `requirements.txt` en Python
        (una dependencia por línea, opcionalmente con versión).
    """
    # Plantilla vacía: implementa aquí tu algoritmo.
    return []
