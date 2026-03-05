from __future__ import annotations

import time
import random
from typing import List, Optional


def solve(problem, time_limit_s: float, seed: int | None = None) -> List[List[str]]:
    """
    Ejemplo simple basado en la plantilla `algoritmoSmartEcoRutas.py`.

    Misma firma que usa run.py:
        solve(problem, time_limit_s=<segundos>, seed=<entero o None>)

    Idea didáctica:
      - Construye rutas secuencialmente.
      - En cada paso toma el contenedor factible más cercano no visitado.
      - Si se llena la capacidad entre DUMPs, visita DUMP y sigue.
      - Siempre intenta mantener margen para cerrar en ... DUMP -> BASE.
      - Las rutas resultantes NO pretenden ser buenas: es solo una referencia simple.
    """
    t0 = time.time()
    rng = random.Random(0 if seed is None else int(seed))

    base = problem.base_uid()
    dump = problem.dump_uid()
    cap = int(problem.max_containers_before_dump)
    route_max = float(problem.route_max_work_s)
    svc_c = float(problem.service_time_container_s)
    svc_d = float(problem.service_time_dump_s)

    all_conts = list(problem.containers_uids())
    if not all_conts:
        return [[base, dump, base]]
    all_conts_set = set(all_conts)
    unvisited = set(all_conts)

    soft_deadline = t0 + 0.96 * max(0.0, float(time_limit_s))
    hard_deadline = t0 + max(0.0, float(time_limit_s))

    def near_limit() -> bool:
        return time.time() >= soft_deadline

    def hard_limit() -> bool:
        return time.time() >= hard_deadline

    def can_still_close_after_visit(cur: str, nxt: str, elapsed_s: float) -> bool:
        projected = elapsed_s + float(problem.time_uid(cur, nxt)) + svc_c
        close_tail = float(problem.time_uid(nxt, dump)) + svc_d + float(problem.time_uid(dump, base))
        return (projected + close_tail) <= route_max + 1e-9

    def nearest_feasible(cur: str, elapsed_s: float) -> Optional[str]:
        if not unvisited:
            return None
        exclude = all_conts_set - unvisited
        k = min(80, len(unvisited))
        cands = problem.k_nearest(cur, k, only_containers=True, exclude=exclude)
        for uid, _ in cands:
            if uid in unvisited and can_still_close_after_visit(cur, uid, elapsed_s):
                return uid
        return None

    routes: List[List[str]] = []
    route_idx = 0
    while unvisited and not hard_limit():
        route_idx += 1
        route = [base]
        cur = base
        load = 0
        elapsed = 0.0
        collected = 0

        while unvisited and not near_limit():
            if load >= cap:
                to_dump = float(problem.time_uid(cur, dump))
                if elapsed + to_dump + svc_d + float(problem.time_uid(dump, base)) > route_max:
                    break
                route.append(dump)
                elapsed += to_dump + svc_d
                cur = dump
                load = 0
                continue

            nxt = nearest_feasible(cur, elapsed)
            if nxt is None:
                break

            route.append(nxt)
            elapsed += float(problem.time_uid(cur, nxt)) + svc_c
            cur = nxt
            load += 1
            collected += 1
            unvisited.remove(nxt)

        if collected == 0 and unvisited:
            pick = min(unvisited, key=lambda u: problem.time_uid(base, u))
            route = [base, pick]
            unvisited.remove(pick)

        if route[-1] != dump:
            route.append(dump)
        if route[-1] != base:
            route.append(base)
        routes.append(route)

        n_cont = sum(1 for u in route if problem.is_container(u))
        n_dump_visits = sum(1 for u in route if u == dump)
        replenish_extra = max(0, n_dump_visits - 1)
        travel_s = float(problem.travel_time_route_uids(route))
        total_s = float(problem.total_time_route_uids(route))
        print(
            f"[SIMPLE-EXAMPLE] ruta {route_idx} | contenedores={n_cont} | travel_h={travel_s/3600.0:.2f} | "
            f"total_h={total_s/3600.0:.2f} | visitas_dump={n_dump_visits} | "
            f"replenish_extra={replenish_extra} | restantes={len(unvisited)}",
            flush=True,
        )

    if unvisited:
        rest = list(unvisited)
        rng.shuffle(rest)
        for u in rest:
            routes.append([base, u, dump, base])

    return routes
