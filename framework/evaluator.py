from __future__ import annotations

"""
Evaluador oficial de soluciones (capa de validación y métricas).

Pensado para alumnos:
  - Aquí se comprueba si las rutas devueltas por `solve(...)` cumplen formato
    y restricciones duras (cierres, tiempos, capacidad, cobertura, duplicados).
  - También se calculan métricas que luego verás en `run.py` y `report.json`.

Nota: este archivo NO ejecuta heurísticos; solo evalúa lo que devuelve tu algoritmo.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
from collections import Counter

from framework.problem_instance import ProblemInstance


@dataclass
class EvalResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


# ------------------------------
# Console spam guards
# ------------------------------
MAX_ERRORS_TOTAL = 80
MAX_WARNINGS_TOTAL = 80
MAX_ERRORS_PER_ROUTE = 10
MAX_WARNINGS_PER_ROUTE = 6
MAX_UNKNOWN_UIDS_PER_ROUTE = 6

# New: per-route notes truncation for route stats
MAX_ROUTE_NOTES = 6


# ------------------------------
# Formatting helpers
# ------------------------------
def _h(seconds: float) -> float:
    """Seconds -> hours (float)."""
    return float(seconds) / 3600.0


def _fmt_h(seconds: float) -> str:
    """Seconds -> 'X.XXh'."""
    return f"{_h(seconds):.2f}h"

def _ceil_div(a: int, b: int) -> int:
    if b <= 0:
        return 0
    return (a + b - 1) // b

def _is_list_of_list_of_str(routes: Any) -> bool:
    if not isinstance(routes, list):
        return False
    for r in routes:
        if not isinstance(r, list):
            return False
        for u in r:
            if not isinstance(u, str):
                return False
    return True


def _validate_route_convention(problem: ProblemInstance, route: Sequence[str]) -> List[str]:
    """
    Enunciado (fijo):
      - Cada ruta empieza en BASE
      - Puede visitar DUMP múltiples veces (replenishment)
      - Termina en BASE y el penúltimo nodo debe ser DUMP  => ... DUMP, BASE
    """
    errs: List[str] = []
    if len(route) < 2:
        errs.append("Ruta demasiado corta. Debe empezar en BASE y terminar en ... DUMP, BASE.")
        return errs

    base = problem.base_uid()
    dump = problem.dump_uid()

    if route[0] != base:
        errs.append(f"No empieza en BASE: route[0]='{route[0]}' (esperado '{base}').")

    if route[-1] != base:
        errs.append(f"No termina en BASE: route[-1]='{route[-1]}' (esperado '{base}').")

    if len(route) >= 2 and route[-2] != dump:
        errs.append(f"Debe terminar en ... DUMP, BASE: penúltimo='{route[-2]}' (esperado '{dump}').")

    return errs


def _validate_uids_exist(problem: ProblemInstance, route: Sequence[str]) -> List[str]:
    errs: List[str] = []
    unknown = 0
    for u in route:
        try:
            problem.uid_to_index(u)
        except KeyError:
            unknown += 1
            if unknown <= MAX_UNKNOWN_UIDS_PER_ROUTE:
                errs.append(f"UID desconocido: '{u}' (no existe en nodes.csv).")
    if unknown > MAX_UNKNOWN_UIDS_PER_ROUTE:
        errs.append(f"... y {unknown - MAX_UNKNOWN_UIDS_PER_ROUTE} UID(s) desconocidos más en esta ruta.")
    return errs


def _route_time_capacity_and_worst_segment(
    problem: ProblemInstance, route: Sequence[str]
) -> Tuple[float, float, float, int, int, int, int]:
    """
    Devuelve:
      - t_total (travel + service) [s]
      - t_travel [s]
      - t_service [s]
      - max_consecutive_containers_between_dumps
      - number_of_dump_visits_in_route
      - worst_segment_index (0-based, entre dumps; el último tramo antes del final también cuenta)
      - worst_segment_containers (contenedores en ese tramo)
    """
    dump = problem.dump_uid()
    t_total = float(problem.total_time_route_uids(route))
    t_travel = float(problem.travel_time_route_uids(route))
    t_service = max(0.0, t_total - t_travel)

    seg_idx = 0
    curr = 0
    dump_visits = 0
    max_cap = 0
    worst_seg_idx = 0
    worst_seg_val = 0

    for u in route:
        if u == dump:
            dump_visits += 1
            if curr > max_cap:
                max_cap = curr
                worst_seg_idx = seg_idx
                worst_seg_val = curr
            curr = 0
            seg_idx += 1
        else:
            try:
                if problem.is_container(u):
                    curr += 1
            except KeyError:
                pass

    if curr > max_cap:
        max_cap = curr
        worst_seg_idx = seg_idx
        worst_seg_val = curr

    return (
        t_total,
        t_travel,
        t_service,
        int(max_cap),
        int(dump_visits),
        int(worst_seg_idx),
        int(worst_seg_val),
    )


def evaluate_solution(
    problem: ProblemInstance,
    routes: Any,
    *,
    compute_time_s: float,
    time_limit_s: float,
) -> EvalResult:
    """
    Valida la solución del alumno y devuelve:
      - ok: True/False (si cumple restricciones duras)
      - errors / warnings: mensajes humanos (limitados para no saturar)
      - stats: métricas y detalles para report.json

    Cambios solicitados:
      - NO imprimir/listar por pantalla los nombres de contenedores NO visitados (solo número).
      - Si algún contenedor se visita 2+ veces => ERROR (se indica como error; en stats se guardan ejemplos acotados).
      - Incluir listado de rutas en stats con: tiempo total, travel, #contenedores, #visitas a DUMP, ok por ruta, notes.
    """
    errors: List[str] = []
    warnings: List[str] = []

    def add_error(msg: str):
        if len(errors) < MAX_ERRORS_TOTAL:
            errors.append(msg)

    def add_warning(msg: str):
        if len(warnings) < MAX_WARNINGS_TOTAL:
            warnings.append(msg)

    if not _is_list_of_list_of_str(routes):
        return EvalResult(
            ok=False,
            errors=[
                "Salida inválida: se esperaba list[list[str]].",
                "Ejemplo: [['BASE','c1','c2','DUMP','BASE'], ['BASE','c3','DUMP','BASE']].",
                "Nota: cada ruta debe empezar en BASE y terminar en ... DUMP, BASE; puede haber contenedores y DUMPs intermedios.",
            ],
            warnings=[],
            stats={"compute_time_s": float(compute_time_s), "time_limit_s": float(time_limit_s)},
        )

    base = problem.base_uid()
    dump = problem.dump_uid()

    # max_routes se trata como referencia (no límite estricto de factibilidad)
    if len(routes) > problem.max_routes:
        add_warning(
            f"Rutas por encima de la referencia: {len(routes)} > max_routes={problem.max_routes}."
        )

    route_times_s: List[float] = []
    route_travel_times_s: List[float] = []
    route_service_times_s: List[float] = []
    route_max_consecutive_containers: List[int] = []
    route_dump_visits: List[int] = []
    worst_capacity_segments: List[Dict[str, Any]] = []

    all_container_visits: List[str] = []

    # New: per-route detail objects
    routes_detail: List[Dict[str, Any]] = []

    for ridx, route in enumerate(routes, start=1):
        route_errs = 0
        route_warns = 0

        # Collect per-route notes (compact)
        route_notes: List[str] = []
        route_errors_list: List[str] = []

        def note(msg: str):
            if len(route_notes) < MAX_ROUTE_NOTES:
                route_notes.append(msg)

        def route_error(msg: str):
            nonlocal route_errs
            if route_errs < MAX_ERRORS_PER_ROUTE:
                add_error(f"[Ruta {ridx}] {msg}")
                route_errors_list.append(msg)
                route_errs += 1
                note(f"ERR: {msg}")

        def route_warning(msg: str):
            nonlocal route_warns
            if route_warns < MAX_WARNINGS_PER_ROUTE:
                add_warning(f"[Ruta {ridx}] {msg}")
                route_warns += 1
                note(f"WARN: {msg}")

        # 1) Convención BASE...DUMP,BASE (ERROR)
        for e in _validate_route_convention(problem, route):
            route_error(e)

        # 2) UIDs existen (ERROR)
        for e in _validate_uids_exist(problem, route):
            route_error(e)

        # 3) Warnings estructurales (ayuda)
        if len(route) > 2 and base in route[1:-1]:
            route_warning(
                f"BASE aparece en medio (uid='{base}'). Normalmente indica un bug: BASE solo debería estar al inicio y al final."
            )

        if route and route[0] == dump:
            route_warning(
                f"La ruta empieza en DUMP. El enunciado exige empezar en BASE (uid='{base}')."
            )

        # 4) Tiempo y capacidad (con segmento peor)
        t_total_s, t_travel_s, t_service_s, maxcap, dump_visits, worst_seg_idx, worst_seg_val = (
            _route_time_capacity_and_worst_segment(problem, route)
        )
        route_times_s.append(float(t_total_s))
        route_travel_times_s.append(float(t_travel_s))
        route_service_times_s.append(float(t_service_s))
        route_max_consecutive_containers.append(int(maxcap))
        route_dump_visits.append(int(dump_visits))
        worst_capacity_segments.append(
            {
                "route_index": ridx,
                "worst_segment_index": worst_seg_idx,
                "worst_segment_containers": worst_seg_val,
            }
        )

        # Restricción jornada (ERROR) — mostrar en horas
        if t_total_s > problem.route_max_work_s:
            route_error(
                f"Excede tiempo máximo: {_fmt_h(t_total_s)} > {_fmt_h(problem.route_max_work_s)}. "
                "Sugerencias: (1) reparte contenedores en más rutas, (2) inserta DUMP antes, (3) reordena para reducir viaje."
            )

        # Restricción capacidad (ERROR)
        if maxcap > problem.max_containers_before_dump:
            route_error(
                f"Excede capacidad del camión entre dos visitas a DUMP: {maxcap} > {problem.max_containers_before_dump}. "
                f"Peor tramo = segmento #{worst_seg_idx} con {worst_seg_val} contenedores. "
                "Sugerencias: visita DUMP más a menudo o reparte en más rutas."
            )

        # 5) Contenedores visitados (robusto ante UID desconocido)
        route_containers: List[str] = []
        for u in route:
            try:
                if problem.is_container(u):
                    all_container_visits.append(u)
                    route_containers.append(u)
            except KeyError:
                pass

        # Duplicados dentro de ruta => ERROR (lo pediste como error si se visita dos veces)
        c_route = Counter(route_containers)
        local_dupes = sorted([u for u, c in c_route.items() if c > 1])
        if local_dupes:
            # No queremos flood: solo mostramos algunos ejemplos aquí (aún así no son "missing")
            ex = local_dupes[:10]
            route_error(
                f"Hay contenedores repetidos dentro de la ruta: {len(local_dupes)} (ej: {ex}). "
                "Un contenedor debe recogerse una sola vez."
            )

        if route_errs >= MAX_ERRORS_PER_ROUTE:
            add_warning(f"[Ruta {ridx}] Errores truncados (mostrando solo {MAX_ERRORS_PER_ROUTE}).")
            note(f"WARN: Errores truncados (solo {MAX_ERRORS_PER_ROUTE}).")
        if route_warns >= MAX_WARNINGS_PER_ROUTE:
            add_warning(f"[Ruta {ridx}] Avisos truncados (mostrando solo {MAX_WARNINGS_PER_ROUTE}).")
            note(f"WARN: Avisos truncados (solo {MAX_WARNINGS_PER_ROUTE}).")

        # New: per-route detail for external printing / report.json
        # Nota: n_containers = únicos en esa ruta
        n_cont_unique = len(set(route_containers))
        route_ok = (len([e for e in route_errors_list]) == 0)

        routes_detail.append(
            {
                "route_index": ridx,
                "ok": bool(route_ok),
                "total_time_s": float(t_total_s),
                "total_time_h": _h(t_total_s),
                "travel_time_s": float(t_travel_s),
                "travel_time_h": _h(t_travel_s),
                "service_time_s": float(t_service_s),
                "service_time_h": _h(t_service_s),
                "n_containers": int(n_cont_unique),
                "n_containers_raw": int(len(route_containers)),
                "n_dump_visits": int(dump_visits),
                "max_consecutive_containers_between_dumps": int(maxcap),
                "worst_segment_index": int(worst_seg_idx),
                "worst_segment_containers": int(worst_seg_val),
                "notes": route_notes[:MAX_ROUTE_NOTES],
            }
        )

    # Cobertura y duplicados globales (ERROR)
    containers = set(problem.containers_uids())
    visited = all_container_visits
    visited_set = set(visited)

    missing = sorted(list(containers - visited_set))

    counts = Counter(visited)
    dupes = sorted([u for u, c in counts.items() if c > 1])

    # IMPORTANT: missing => NO listar IDs en errors (solo número)
    if missing:
        add_error(f"Contenedores NO visitados: {len(missing)} (de {len(containers)}).")

    # Global duplicates => ERROR (puede listar ejemplos; esto NO son 'missing')
    if dupes:
        add_error(f"Contenedores visitados más de una vez: {len(dupes)} (ej: {dupes[:10]}).")

    # Métricas globales
    total_time_s = float(sum(route_times_s)) if route_times_s else 0.0
    total_travel_time_s = float(sum(route_travel_times_s)) if route_travel_times_s else 0.0
    makespan_s = float(max(route_times_s)) if route_times_s else 0.0
    coverage = float(len(containers) - len(missing)) / float(len(containers)) if containers else 1.0

    # Ruta más larga / corta (en segundos y en horas)
    route_longest_s = float(max(route_times_s)) if route_times_s else 0.0
    route_shortest_s = float(min(route_times_s)) if route_times_s else 0.0

    # ------------------------------
    # NEW: Dump-visits lower bound (capacity only)
    # ------------------------------
    cap = int(problem.max_containers_before_dump)

    n_total = int(len(containers))
    n_visited_unique = int(len(visited_set))

    dump_visits_total = int(sum(route_dump_visits)) if route_dump_visits else 0

    # Lower bound purely from capacity
    min_dump_visits_all = int(_ceil_div(n_total, cap)) if n_total > 0 else 0
    min_dump_visits_served = int(_ceil_div(n_visited_unique, cap)) if n_visited_unique > 0 else 0

    extra_vs_lb_all = int(dump_visits_total - min_dump_visits_all)
    extra_vs_lb_served = int(dump_visits_total - min_dump_visits_served)

    # Print to console (as requested)
    print(
        "[EVAL] DUMP visits (capacity-based LB only):"
        f" actual={dump_visits_total}"
        f" | cap={cap}"
        f" | N_total={n_total}"
        f" | LB_total=ceil(N_total/cap)={min_dump_visits_all}"
        f" (extra={extra_vs_lb_all:+d})"
        f" | N_served_unique={n_visited_unique}"
        f" | LB_served=ceil(N_served/cap)={min_dump_visits_served}"
        f" (extra={extra_vs_lb_served:+d})"
    )


    if compute_time_s > time_limit_s:
        add_warning(
            f"Tiempo de cómputo supera el límite: {_fmt_h(compute_time_s)} > {_fmt_h(time_limit_s)} "
            f"({compute_time_s:.2f}s > {time_limit_s:.2f}s). "
            "Sugerencia: añade early-stop por tiempo y devuelve la mejor solución conocida."
        )

    ok = len(errors) == 0

    # Nota de truncado global
    if len(errors) >= MAX_ERRORS_TOTAL:
        errors = errors[:MAX_ERRORS_TOTAL] + [f"... (se han omitido más errores; límite={MAX_ERRORS_TOTAL})."]
    if len(warnings) >= MAX_WARNINGS_TOTAL:
        warnings = warnings[:MAX_WARNINGS_TOTAL] + [f"... (se han omitido más avisos; límite={MAX_WARNINGS_TOTAL})."]

    stats = {
        "subproblem": problem.subproblem,
        "ok": ok,

        # tiempos de ejecución del algoritmo
        "compute_time_s": float(compute_time_s),
        "compute_time_h": _h(compute_time_s),
        "time_limit_s": float(time_limit_s),
        "time_limit_h": _h(time_limit_s),

        # tamaño solución
        "n_routes": int(len(routes)),
        "max_routes": int(problem.max_routes),

        # cobertura / duplicados
        "coverage": float(coverage),
        "n_containers_total": int(len(containers)),
        "n_containers_visited_unique": int(len(visited_set)),
        "n_containers_missing": int(len(missing)),          # <- lo que quieres en summary (nº no recogidos)
        "n_containers_duplicates": int(len(dupes)),         # <- nº duplicados globales


        # NEW: DUMP visits global + capacity lower bounds
        "dump_visits_total": int(dump_visits_total),

        "dump_visits_min_lb_all": int(min_dump_visits_all),
        "dump_visits_extra_vs_lb_all": int(extra_vs_lb_all),

        "dump_visits_min_lb_served": int(min_dump_visits_served),
        "dump_visits_extra_vs_lb_served": int(extra_vs_lb_served),
        # tiempos por ruta (total y travel)
        "route_times_s": [float(x) for x in route_times_s],
        "route_times_h": [_h(x) for x in route_times_s],
        "route_travel_times_s": [float(x) for x in route_travel_times_s],
        "route_travel_times_h": [_h(x) for x in route_travel_times_s],
        "route_service_times_s": [float(x) for x in route_service_times_s],
        "route_service_times_h": [_h(x) for x in route_service_times_s],

        # agregados
        "total_time_s": float(total_time_s),
        "total_time_h": _h(total_time_s),
        "total_travel_time_s": float(total_travel_time_s),
        "total_travel_time_h": _h(total_travel_time_s),

        "makespan_s": float(makespan_s),
        "makespan_h": _h(makespan_s),

        "route_longest_s": float(route_longest_s),
        "route_longest_h": _h(route_longest_s),
        "route_shortest_s": float(route_shortest_s),
        "route_shortest_h": _h(route_shortest_s),

        # capacidad / dumps
        "route_max_consecutive_containers": [int(x) for x in route_max_consecutive_containers],
        "route_dump_visits": [int(x) for x in route_dump_visits],
        "worst_capacity_segments": worst_capacity_segments,

        "capacity_limit": int(problem.max_containers_before_dump),
        "route_max_work_s": int(problem.route_max_work_s),
        "route_max_work_h": _h(problem.route_max_work_s),

        # service times
        "service_time_container_s": float(problem.service_time_container_s),
        "service_time_dump_s": float(problem.service_time_dump_s),

        # NEW: detalle por ruta (para que run.py lo liste como quieras)
        "routes": routes_detail,

        # ejemplos (acotados) - OJO: missing_examples sigue estando en report.json, pero run.py NO debe imprimirlo
        "missing_examples": missing[:50],
        "duplicates_examples": dupes[:50],
    }

    return EvalResult(ok=ok, errors=errors, warnings=warnings, stats=stats)
