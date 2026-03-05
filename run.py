from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from framework.problem_instance import ProblemInstance
from framework.evaluator import evaluate_solution
from framework.geo_export import export_for_qgis

# =============================================================================
# Runner para alumnos
# =============================================================================
# Flujo general:
#   1) Carga instancia(s) y módulo del alumno (solve).
#   2) Ejecuta solve(problem, time_limit_s, seed).
#   3) Evalúa restricciones y métricas.
#   4) Guarda report.json y, opcionalmente, export visual (KMZ/GPKG).
#
# Este archivo es intencionalmente explícito para que podáis entender
# qué espera el sistema automático en cada fase.

DEFAULT_INSTANCES = [
    "LATERAL_CARTON",
    "LATERAL_ENVASE",
    "LATERAL_RESTO",
    "TRASERA_RESTO",
]


def _h(seconds: float) -> float:
    return float(seconds) / 3600.0


def _fmt_h(seconds: float) -> str:
    return f"{_h(seconds):.2f}h"


def _pretty_header(title: str) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def _load_student_algorithm(module_path: str):
    """
    Loads student algorithm module.

    Expected file (default):
        student/algoritmoSmartEcoRutas.py

    Expected function:
        solve(problem: ProblemInstance, time_limit_s: float, seed: int | None = None) -> list[list[str]]
    """
    # 1) Ensure project root is on sys.path
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    expected_file = root / "student" / "algoritmoSmartEcoRutas.py"
    expected_module = "student.algoritmoSmartEcoRutas"

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        msg = (
            f"\n[ERROR] No se encontró el algoritmo del alumno.\n"
            f"  - Módulo solicitado: '{module_path}'\n"
            f"  - Módulo por defecto: '{expected_module}'\n\n"
            f"Se esperaba encontrar el fichero:\n"
            f"  {expected_file}\n\n"
            f"Solución:\n"
            f"  1) Crea el directorio 'student/' si no existe.\n"
            f"  2) Añade el fichero 'algoritmoSmartEcoRutas.py' dentro.\n"
            f"  3) Implementa la función obligatoria solve(...).\n\n"
            f"Plantilla mínima (solo formato; NO es una buena solución):\n"
            f"  def solve(problem, time_limit_s: float, seed=None):\n"
            f"      \"\"\"Devuelve list[list[str]]. Cada ruta:\n"
            f"      - empieza en BASE\n"
            f"      - visita contenedores (uids) y puede visitar DUMP varias veces\n"
            f"      - termina en ... DUMP, BASE\n"
            f"      \"\"\"\n"
            f"      base = problem.base_uid()\n"
            f"      dump = problem.dump_uid()\n"
            f"      conts = problem.containers_uids()\n"
            f"      if not conts:\n"
            f"          return [[base, dump, base]]\n"
            f"      # Ejemplo: BASE -> primer contenedor -> DUMP -> BASE\n"
            f"      return [[base, conts[0], dump, base]]\n\n"
            f"Ejemplo con varios contenedores y un replenishment extra:\n"
            f"  [[ 'BASE', 'c_001', 'c_002', 'DUMP', 'c_010', 'DUMP', 'BASE' ]]\n\n"
            f"Detalle técnico: {repr(e)}\n"
        )
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = (
            f"\n[ERROR] No se pudo importar el módulo del alumno '{module_path}'.\n"
            f"Esto suele indicar un error de sintaxis o una excepción al importar.\n\n"
            f"  - Fichero esperado (por defecto): {expected_file}\n"
            f"  - Revisa que el archivo existe y se puede importar.\n"
            f"  - Prueba: python -c \"import {module_path}\"\n\n"
            f"Detalle técnico: {repr(e)}\n"
        )
        raise RuntimeError(msg) from e

    # Must define solve
    if not hasattr(mod, "solve"):
        msg = (
            f"\n[ERROR] El módulo '{module_path}' se importó, pero NO define la función obligatoria solve(...).\n\n"
            f"Debes implementar EXACTAMENTE:\n"
            f"  def solve(problem, time_limit_s: float, seed: int | None = None) -> list[list[str]]:\n"
            f"      ...\n\n"
            f"Fichero esperado (por defecto):\n"
            f"  {expected_file}\n"
        )
        raise RuntimeError(msg)

    # Lightweight signature feedback
    import inspect

    sig = inspect.signature(mod.solve)
    params = list(sig.parameters.keys())
    if len(params) < 2:
        msg = (
            f"\n[ERROR] La firma de solve(...) no es correcta.\n"
            f"Firma encontrada: solve{sig}\n\n"
            f"Firma requerida:\n"
            f"  solve(problem, time_limit_s: float, seed: int | None = None)\n\n"
            f"Ejemplo:\n"
            f"  def solve(problem, time_limit_s: float, seed=None):\n"
            f"      return [[problem.base_uid(), problem.dump_uid(), problem.base_uid()]]\n"
        )
        raise RuntimeError(msg)

    time_like = any("time" in p.lower() for p in params)
    if not time_like:
        print(
            f"[WARN] solve(...) no tiene un parámetro con nombre parecido a 'time_limit_s'. "
            f"Se intentará llamar por posición. Firma encontrada: solve{sig}"
        )

    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmartEcoRutas StudentKit runner: ejecuta el algoritmo del alumno sobre una o varias instancias."
    )
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directorio con las instancias (por defecto: student-kit/data).",
    )
    p.add_argument(
        "--out-dir",
        default="algorithm_output",
        help="Directorio donde se guardan report.json y (opcional) export visual KMZ/GPKG.",
    )
    p.add_argument(
        "--instances",
        nargs="*",
        default=DEFAULT_INSTANCES,
        help=f"Lista de instancias a ejecutar (por defecto: {', '.join(DEFAULT_INSTANCES)}).",
    )
    p.add_argument(
        "--time-limit-min",
        type=float,
        default=15.0,
        help="Límite de tiempo (minutos) que se pasa al algoritmo (por instancia). Por defecto 15.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Semilla (seed) para reproducibilidad. 0 por defecto.",
    )
    p.add_argument(
        "--no-geo",
        action="store_true",
        help="Si se activa, NO genera ficheros de visualización (KMZ/GPKG).",
    )
    p.add_argument(
        "--algo-module",
        default="student.algoritmoSmartEcoRutas",
        help="Ruta del módulo del algoritmo del alumno (por defecto: student.algoritmoSmartEcoRutas).",
    )
    p.add_argument(
        "--use-simple-example",
        action="store_true",
        help="Usa el algoritmo didáctico: student.algoritmoSmartEcoRutas_simple_example.",
    )
    return p.parse_args()


def _save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _print_routes_table(eval_stats: dict) -> None:
    """
    Imprime una tabla compacta de rutas usando stats["routes"] (generado por evaluator nuevo).
    No imprime UIDs de contenedores NO visitados (eso va agregado como número a nivel global).
    """
    routes = eval_stats.get("routes", []) or []
    if not routes:
        print("[ROUTES] (sin detalle por ruta en stats)")
        return

    print("\n[ROUTES resumen]")
    header = "  idx | ok | total(h) | travel(h) | service(h) | contenedores | visitas_dump | max_cont_entre_dump | notes"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in routes:
        idx = int(r.get("route_index", -1))
        ok = bool(r.get("ok", False))

        total_h = float(r.get("total_time_h", 0.0))
        travel_h = float(r.get("travel_time_h", 0.0))
        service_h = float(r.get("service_time_h", 0.0))

        n_cont = int(r.get("n_containers", 0))
        n_dump = int(r.get("n_dump_visits", 0))
        maxcap = int(r.get("max_consecutive_containers_between_dumps", 0))

        notes_list = r.get("notes", []) or []
        notes = notes_list[0] if notes_list else ""

        # recorta notas muy largas para no desbordar
        if len(notes) > 120:
            notes = notes[:117] + "..."

        print(
            f"  {idx:>3} | {'Y' if ok else 'N'}  | {total_h:>7.2f} | {travel_h:>8.2f} | {service_h:>9.2f} "
            f"| {n_cont:>12} | {n_dump:>12} | {maxcap:>18} | {notes}"
        )


def main() -> int:
    args = _parse_args()
    if args.use_simple_example:
        args.algo_module = "student.algoritmoSmartEcoRutas_simple_example"

    root = Path(__file__).resolve().parent
    data_dir = (root / args.data_dir).resolve()
    out_dir = (root / args.out_dir).resolve()

    time_limit_s = float(args.time_limit_min) * 60.0
    seed = int(args.seed) if args.seed is not None else None

    _pretty_header("SmartEcoRutas StudentKit - RUN")
    print(f"[CONFIG] data_dir   : {data_dir}")
    print(f"[CONFIG] out_dir    : {out_dir}")
    print(f"[CONFIG] instances  : {args.instances}")
    print(
        f"[CONFIG] time_limit : {args.time_limit_min:.2f} min ({time_limit_s:.0f} s = {_fmt_h(time_limit_s)}) | seed={seed}"
    )
    print(f"[CONFIG] export_geo : {'OFF' if args.no_geo else 'ON'}")
    print(f"[CONFIG] algoritmo  : {args.algo_module}")

    # 1) Import student algorithm
    try:
        algo = _load_student_algorithm(args.algo_module)
    except RuntimeError as e:
        print(str(e))
        return 2

    print(f"[OK] Algoritmo cargado: {args.algo_module}.solve(...)")

    # 2) Loop instances
    global_ok = True
    summary_rows = []

    for inst in args.instances:
        _pretty_header(f"[INSTANCE] {inst}")

        inst_dir = data_dir / inst
        try:
            problem = ProblemInstance.load_from_dir(inst_dir)
            print(f"[OK] Input cargado: K={problem.K} (BASE + DUMP + contenedores={problem.container_count})")
            print(
                f"     max_routes_ref={problem.max_routes} | capacidad_camion_cont={problem.max_containers_before_dump} "
                f"| tiempo_max_ruta={problem.route_max_work_s}s ({_fmt_h(problem.route_max_work_s)})"
            )
            print(
                f"     tiempo_servicio: contenedor={problem.service_time_container_s:.1f}s "
                f"dump={problem.service_time_dump_s:.1f}s"
            )
            print("     objetivo: minimizar primero #rutas, luego tiempo total (max_routes es referencia)")
        except Exception as e:
            global_ok = False
            print("[ERROR] Fallo cargando input.")
            print(f"        Instancia: {inst_dir}")
            print("        Deben existir (según instance.json/files): instance.json, nodes.csv y time_matrix.npz")
            print("        Sugerencia: re-genera el dataset con NB13 y revisa que copiaste bien data/<instancia>/")
            print(f"        Detalle: {repr(e)}")
            continue

        # 3) Execute algorithm
        print("\n[RUN] Ejecutando algoritmo del alumno...")
        t0 = time.time()
        try:
            routes = algo.solve(problem, time_limit_s=time_limit_s, seed=seed)
        except TypeError:
            # If student did not accept kwargs, try positional
            routes = algo.solve(problem, time_limit_s, seed)
        except Exception as e:
            global_ok = False
            print("[ERROR] El algoritmo lanzó una excepción durante la ejecución.")
            print("        Esto suele pasar por índices fuera de rango, uids mal usados, o estructuras de salida incorrectas.")
            print(f"        Detalle: {repr(e)}")
            _save_json(
                out_dir / inst / "report.json",
                {
                    "schema_version": 1,
                    "instance": inst,
                    "ok": False,
                    "algorithm_module": args.algo_module,
                    "runtime_error": repr(e),
                },
            )
            continue
        t1 = time.time()
        compute_time_s = t1 - t0
        print(f"[OK] Algoritmo terminó. compute_time={compute_time_s:.2f}s ({_fmt_h(compute_time_s)})")

        # 4) Evaluate solution
        print("\n[EVAL] Validando solución...")
        eval_res = evaluate_solution(
            problem,
            routes,
            compute_time_s=compute_time_s,
            time_limit_s=time_limit_s,
        )

        # Print errors/warnings (already truncated by evaluator, but we still avoid flooding)
        if eval_res.errors:
            print("\n[❌ ERRORES (restricciones NO cumplidas)]")
            for e in eval_res.errors[:50]:
                print(" -", e)
            if len(eval_res.errors) > 50:
                print(f" ... ({len(eval_res.errors) - 50} errores más)")
        else:
            print("[✅] No hay errores de restricciones.")

        display_warnings = [
            w for w in eval_res.warnings
            if not str(w).startswith("Rutas por encima de la referencia:")
        ]
        if display_warnings:
            print("\n[⚠️ WARNINGS (pistas / mejoras)]")
            for w in display_warnings[:50]:
                print(" -", w)
            if len(display_warnings) > 50:
                print(f" ... ({len(display_warnings) - 50} warnings más)")
        else:
            print("[OK] Sin warnings.")

        # NEW: per-route table
        _print_routes_table(eval_res.stats)

        # 5) Build report.json
        report = {
            "schema_version": 1,
            "instance": inst,
            "algorithm_module": args.algo_module,
            "evaluation": eval_res.stats,
            "errors": eval_res.errors,
            "warnings": eval_res.warnings,
        }

        # 6) Optional: export visual (KMZ + GPKG)
        if not args.no_geo:
            try:
                nb11_assets_dir = root / "data" / "nb11_outputs"
                nb11_assets_dir.mkdir(parents=True, exist_ok=True)
                geo_info = export_for_qgis(
                    problem,
                    routes,
                    out_dir / inst,
                    nb11_tag=inst,
                    repo_root=root,  # root = student-kit/
                    nb11_outputs_dir="data/nb11_outputs",  # <- aquí están tus ficheros
                )
                report["geo_export"] = geo_info
                print("[OK] Export visual generado (KMZ/GPKG):", geo_info)
            except Exception as e:
                report["geo_export_error"] = repr(e)
                print("[ERROR GRAVE] No se pudo generar export visual (KMZ/GPKG):", repr(e))
                _save_json(out_dir / inst / "report.json", report)
                print(f"[SAVE] report.json -> {out_dir / inst / 'report.json'}")
                return 2

        _save_json(out_dir / inst / "report.json", report)
        print(f"[SAVE] report.json -> {out_dir / inst / 'report.json'}")

        # 7) Friendly summary per instance (hours + travel)
        ok = bool(eval_res.ok)
        global_ok = global_ok and ok

        cov = float(eval_res.stats.get("coverage", 0.0))
        n_routes = int(eval_res.stats.get("n_routes", 0))

        # totals (note: evaluator usa total_travel_time_s / total_time_s)
        total_time_s = float(eval_res.stats.get("total_time_s", 0.0))
        total_travel_s = float(eval_res.stats.get("total_travel_time_s", 0.0))
        makespan_s = float(eval_res.stats.get("makespan_s", 0.0))

        route_longest_s = float(eval_res.stats.get("route_longest_s", 0.0))
        route_shortest_s = float(eval_res.stats.get("route_shortest_s", 0.0))
        max_routes_ref = int(eval_res.stats.get("max_routes", problem.max_routes))
        dump_visits_total = int(eval_res.stats.get("dump_visits_total", 0))
        replenish_extra = max(0, dump_visits_total - n_routes)

        n_missing = int(eval_res.stats.get("n_containers_missing", 0))
        n_dupes = int(eval_res.stats.get("n_containers_duplicates", 0))

        summary_rows.append(
            {
                "instance": inst,
                "ok": ok,
                "compute_time_s": float(compute_time_s),
                "coverage": cov,
                "n_routes": n_routes,
                "n_missing": n_missing,
                "n_dupes": n_dupes,
                "total_time_s": total_time_s,
                "total_travel_s": total_travel_s,
                "makespan_s": makespan_s,
                "route_longest_s": route_longest_s,
                "route_shortest_s": route_shortest_s,
                "max_routes_ref": max_routes_ref,
                "replenish_extra": replenish_extra,
            }
        )

        print("\n[SUMMARY instancia]")
        print(
            f"  ok={ok} | routes={n_routes} | max_routes_ref={max_routes_ref} | "
            f"replenish_extra={replenish_extra}"
        )
        if not ok:
            print(f"  detalle: coverage={cov:.3f} | missing={n_missing} | dupes={n_dupes}")
        print(f"  total_time (travel+service) = {_fmt_h(total_time_s)}  ({total_time_s:.0f}s)")
        print(f"  total_travel_time           = {_fmt_h(total_travel_s)}  ({total_travel_s:.0f}s)")
        print(f"  makespan (ruta más larga)   = {_fmt_h(makespan_s)}  ({makespan_s:.0f}s)")
        print(f"  ruta más larga/corta        = {_fmt_h(route_longest_s)} / {_fmt_h(route_shortest_s)}")
        print(f"  compute_time                = {compute_time_s:.2f}s (limit={time_limit_s:.2f}s)")

    # 8) Global summary
    _pretty_header("[GLOBAL SUMMARY]")
    if not summary_rows:
        print("No se pudo ejecutar ninguna instancia (revisa errores anteriores).")
        return 2

    for r in summary_rows:
        print(
            f"- {r['instance']}: ok={r['ok']} | routes={r['n_routes']} | max_routes_ref={r['max_routes_ref']} "
            f"| replenish_extra={r['replenish_extra']} "
            f"| total={_fmt_h(r['total_time_s'])} | travel={_fmt_h(r['total_travel_s'])} "
            f"| makespan={_fmt_h(r['makespan_s'])} | cpu={_fmt_h(r['compute_time_s'])}"
        )

    print(f"\nSalida guardada en: {out_dir}")
    return 0 if global_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
