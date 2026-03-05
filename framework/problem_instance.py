# framework/problem_instance.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import json
import time
import numpy as np
import pandas as pd


# ------------------------------
# Helpers
# ------------------------------
def _ceil_div(a: int, b: int) -> int:
    """Ceil division for non-negative ints."""
    if b <= 0:
        return 0
    return (a + b - 1) // b


# ------------------------------
# Core data structures
# ------------------------------
@dataclass(frozen=True)
class Node:
    """A node in the instance (BASE, DUMP or a container)."""
    uid: str
    kind: str  # "base" | "dump" | "container"
    lon: float
    lat: float
    x: float
    y: float


class ProblemInstance:
    """
    Loads and provides access to a SmartEcoRutas instance.

    Expected files inside an instance directory:
      - instance.json
      - nodes.csv
      - time_matrix.npz  (with key "T" by default)

    Key design points:
      - The order of rows in nodes.csv defines the index order used by the matrix T.
      - T[i, j] is the travel time in seconds from nodes[i] to nodes[j].
      - Service times are stored in instance.json and are separate from travel times.

    Performance notes:
      - We cache indices (BASE/DUMP/containers) and frequently used arrays.
      - We can precompute sorted neighbor indices per row to make k_nearest() fast.
      - We print timing information for the expensive precomputations.

    Student quick reference (common calls in your algorithm):
      - problem.base_uid() / problem.dump_uid()
      - problem.containers_uids()
      - problem.time_uid(uid_a, uid_b)
      - problem.k_nearest(uid, k, only_containers=True, exclude=...)
      - problem.travel_time_route_uids(route)
      - problem.service_time_route_uids(route)
      - problem.total_time_route_uids(route)

    Typical route format used across the project:
      ["BASE", "c_001", "c_017", "DUMP", "BASE"]
    """

    # ------------------------------
    # Construction / loading
    # ------------------------------
    def __init__(
        self,
        *,
        instance_dir: Path,
        meta: dict,
        nodes: list[Node],
        uid_to_index: dict[str, int],
        T: np.ndarray,
        base_uid: str,
        dump_uid: str,
        precompute_neighbors: bool = True,
        verbose: bool = True,
    ):
        self._instance_dir = instance_dir
        self._meta = meta
        self._nodes = nodes
        self._uid_to_index = uid_to_index
        self._T = T.astype(np.float32, copy=False)

        # Cache: BASE/DUMP uids and indices
        self._base_uid = str(base_uid)
        self._dump_uid = str(dump_uid)
        self._base_index = self._uid_to_index[self._base_uid]
        self._dump_index = self._uid_to_index[self._dump_uid]

        # Cache: uid list in matrix order
        self._uid_list = [n.uid for n in self._nodes]

        # Cache: kind by index (fast checks in hot path)
        self._kind_by_index = [n.kind for n in self._nodes]

        # Cache: container indices and container uids
        self._container_indices = [i for i, k in enumerate(self._kind_by_index) if k == "container"]
        self._container_uids = [self._uid_list[i] for i in self._container_indices]

        # Optional: precompute row-wise neighbor order to speed up k_nearest()
        self._row_sorted_idx: Optional[np.ndarray] = None
        self._neighbor_precompute_s: float = 0.0

        if precompute_neighbors:
            t0 = time.perf_counter()
            self._row_sorted_idx = self._precompute_row_sorted_indices()
            self._neighbor_precompute_s = time.perf_counter() - t0
            if verbose:
                K = self.K
                print(
                    f"[ProblemInstance] Precompute neighbors: K={K} "
                    f"-> {self._neighbor_precompute_s:.3f}s"
                )

        # Quick internal sanity (cheap)
        self._sanity_post_init()

    @staticmethod
    def load_from_dir(
        instance_dir: str | Path,
        *,
        precompute_neighbors: bool = True,
        verbose: bool = True,
    ) -> "ProblemInstance":
        """
        Load an instance from a directory containing:
          - instance.json
          - nodes.csv
          - time_matrix.npz

        This method performs validation checks to avoid silent mismatches.
        """
        instance_dir = Path(instance_dir)
        if not instance_dir.exists():
            raise FileNotFoundError(f"Instance directory not found: {instance_dir}")

        json_path = instance_dir / "instance.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing instance.json in {instance_dir}")

        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        files = meta.get("files", {})
        nodes_name = files.get("nodes", "nodes.csv")
        matrix_name = files.get("time_matrix", "time_matrix.npz")
        matrix_key = files.get("time_matrix_npz_key", "T")

        nodes_path = instance_dir / nodes_name
        matrix_path = instance_dir / matrix_name

        if not nodes_path.exists():
            raise FileNotFoundError(f"Missing nodes file: {nodes_path}")
        if not matrix_path.exists():
            raise FileNotFoundError(f"Missing time matrix file: {matrix_path}")

        # Load nodes.csv (portable; no pyarrow required)
        t_nodes0 = time.perf_counter()
        df = pd.read_csv(nodes_path)
        t_nodes = time.perf_counter() - t_nodes0

        required_cols = ["uid", "kind", "lon", "lat", "x", "y"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"nodes.csv missing required columns: {missing}. Found: {list(df.columns)}")

        nodes: list[Node] = []
        uid_to_index: dict[str, int] = {}

        # itertuples is faster than iterrows
        for row in df.itertuples(index=False):
            uid = str(getattr(row, "uid"))
            kind = str(getattr(row, "kind")).lower().strip()

            if uid in uid_to_index:
                raise ValueError(f"Duplicate uid in nodes.csv: {uid}")

            if kind not in ("base", "dump", "container"):
                raise ValueError(f"Invalid kind='{kind}' for uid={uid}. Must be base|dump|container.")

            node = Node(
                uid=uid,
                kind=kind,
                lon=float(getattr(row, "lon")),
                lat=float(getattr(row, "lat")),
                x=float(getattr(row, "x")),
                y=float(getattr(row, "y")),
            )
            uid_to_index[uid] = len(nodes)
            nodes.append(node)

        # BASE/DUMP ids
        base_uid = files.get("base_uid", "BASE")
        dump_uid = files.get("dump_uid", "DUMP")

        if base_uid not in uid_to_index:
            raise ValueError(f"BASE uid '{base_uid}' not found in nodes.csv")
        if dump_uid not in uid_to_index:
            raise ValueError(f"DUMP uid '{dump_uid}' not found in nodes.csv")

        # Load matrix
        t_mat0 = time.perf_counter()
        T = ProblemInstance._load_time_matrix_npz(matrix_path, key=matrix_key)
        t_mat = time.perf_counter() - t_mat0

        # Validate matrix shape vs nodes
        K = len(nodes)
        if T.shape != (K, K):
            raise ValueError(
                f"Matrix shape {T.shape} does not match number of nodes {K}. "
                f"(nodes.csv order defines matrix indices)"
            )

        # Validate finiteness
        if not np.all(np.isfinite(T)):
            bad = np.argwhere(~np.isfinite(T))
            sample = bad[:10].tolist()
            raise ValueError(f"Time matrix contains non-finite values (inf/nan). Sample indices: {sample}")

        # Diagonal should be 0 (or very close)
        diag = np.diag(T)
        if not np.allclose(diag, 0.0, atol=1e-4):
            raise ValueError("Time matrix diagonal is not (approximately) zero; instance might be corrupted.")

        if verbose:
            print(
                f"[ProblemInstance] Load: nodes.csv {t_nodes:.3f}s | "
                f"matrix {t_mat:.3f}s | K={K}"
            )

        inst = ProblemInstance(
            instance_dir=instance_dir,
            meta=meta,
            nodes=nodes,
            uid_to_index=uid_to_index,
            T=T,
            base_uid=base_uid,
            dump_uid=dump_uid,
            precompute_neighbors=precompute_neighbors,
            verbose=verbose,
        )

        # Print LBs in concise form (details can be documented elsewhere)
        if verbose:
            lb_dumps = inst.min_dump_visits_capacity_lb_total
            lb_routes_time = inst.min_routes_time_lb_service_budget

            print(f"[ProblemInstance] LB visitas a DUMP: {lb_dumps}")
            print(f"[ProblemInstance] LB rutas (por tiempo de servicio): {lb_routes_time}")

        return inst

    @staticmethod
    def _load_time_matrix_npz(path: Path, *, key: str = "T") -> np.ndarray:
        """Load compressed NPZ matrix and return as float32 ndarray."""
        data = np.load(path)
        if key not in data:
            raise ValueError(f"NPZ matrix file {path.name} does not contain key '{key}'. Keys: {list(data.keys())}")
        T = data[key]
        if not isinstance(T, np.ndarray) or T.ndim != 2:
            raise ValueError(f"NPZ key '{key}' must be a 2D numpy array.")
        return T.astype(np.float32, copy=False)

    def _sanity_post_init(self) -> None:
        """Cheap sanity checks to fail fast if something is off."""
        if self._base_uid not in self._uid_to_index:
            raise ValueError("Internal error: base uid not in uid_to_index.")
        if self._dump_uid not in self._uid_to_index:
            raise ValueError("Internal error: dump uid not in uid_to_index.")
        if self._kind_by_index[self._base_index] != "base":
            raise ValueError(f"BASE uid '{self._base_uid}' is not kind='base' in nodes.csv.")
        if self._kind_by_index[self._dump_index] != "dump":
            raise ValueError(f"DUMP uid '{self._dump_uid}' is not kind='dump' in nodes.csv.")

    def _precompute_row_sorted_indices(self) -> np.ndarray:
        """
        Precompute, for each i, the list of indices j sorted by T[i,j] ascending.

        This accelerates k_nearest() and many greedy heuristics.
        """
        T = self._T
        K = T.shape[0]
        order = np.argsort(T, axis=1, kind="quicksort")
        if order.shape != (K, K):
            raise ValueError("Unexpected shape in row argsort precompute.")
        return order

    # ------------------------------
    # Meta / parameters (from instance.json)
    # ------------------------------
    @property
    def instance_dir(self) -> Path:
        return self._instance_dir

    @property
    def subproblem(self) -> str:
        return str(self._meta.get("subproblem", ""))

    @property
    def load_type(self) -> str:
        return str(self._meta.get("load_type", ""))

    @property
    def waste_type(self) -> str:
        return str(self._meta.get("waste_type", ""))

    @property
    def max_routes(self) -> int:
        return int(self._meta["limits"]["max_routes"])

    @property
    def max_containers_before_dump(self) -> int:
        return int(self._meta["limits"]["max_containers_before_dump"])

    @property
    def route_max_work_s(self) -> int:
        return int(self._meta["limits"]["route_max_work_s"])

    @property
    def service_time_container_s(self) -> float:
        return float(self._meta["service_times_s"]["container"])

    @property
    def service_time_dump_s(self) -> float:
        return float(self._meta["service_times_s"]["dump"])

    # ------------------------------
    # Nodes and indices
    # ------------------------------
    @property
    def container_count(self) -> int:
        return len(self._container_indices)

    @property
    def min_dump_visits_capacity_lb_total(self) -> int:
        """
        Lower bound (cota inferior) del número total de visitas a DUMP
        *solo por capacidad*, independiente del algoritmo.

        Con N contenedores y cap=max_containers_before_dump:
            LB = ceil(N / cap)
        """
        n = int(self.container_count)
        cap = int(self.max_containers_before_dump)
        return int(_ceil_div(n, cap)) if n > 0 else 0

    @property
    def min_routes_time_lb_service_budget(self) -> int:
        """
        Lower bound del número de rutas basado SOLO en presupuesto de tiempo para servicio
        (muy optimista: IGNORA el travel salvo el retorno final DUMP->BASE por ruta).

        Definición pedida:

          per_route_budget = route_max_work_s - time(DUMP->BASE)

          svc_cont_total = N * service_time_container_s
          svc_dump_lb    = LB_dumps * service_time_dump_s
          total_service_lb = svc_cont_total + svc_dump_lb

          LB_routes = ceil(total_service_lb / per_route_budget)

        Si per_route_budget <= 0 => devuelve un número "infinito" práctico: max_routes + 1.
        """
        n = int(self.container_count)
        if n <= 0:
            return 0

        t_dump_base = float(self.time_uid(self.dump_uid(), self.base_uid()))
        per_route_budget = float(self.route_max_work_s) - t_dump_base
        if per_route_budget <= 0.0:
            # No hay ni tiempo para "hacer trabajo" tras reservar el retorno DUMP->BASE
            return int(self.max_routes) + 1

        lb_dumps = int(self.min_dump_visits_capacity_lb_total)
        svc_dump_lb = float(lb_dumps) * float(self.service_time_dump_s)
        svc_cont_total = float(n) * float(self.service_time_container_s)
        total_service_lb = svc_dump_lb + svc_cont_total

        return int(_ceil_div(int(np.ceil(total_service_lb)), int(np.ceil(per_route_budget))))

    @property
    def max_collectable_by_capacity(self) -> int:
        """
        Máximo número de contenedores que se pueden recoger *solo por capacidad*
        si se usan como mucho max_routes rutas (ignora límites de tiempo).

            max_collectable = max_routes * cap
        """
        return int(self.max_routes) * int(self.max_containers_before_dump)

    @property
    def K(self) -> int:
        """Total number of nodes (BASE + DUMP + containers)."""
        return len(self._nodes)

    @property
    def nodes(self) -> list[Node]:
        """Ordered list of nodes (order matches matrix indices)."""
        return self._nodes

    def uid_to_index(self, uid: str) -> int:
        """Return matrix index for a uid."""
        try:
            return self._uid_to_index[str(uid)]
        except KeyError as e:
            raise KeyError(f"Unknown uid: {uid}") from e

    def index_to_uid(self, i: int) -> str:
        """Return uid for a matrix index."""
        if i < 0 or i >= self.K:
            raise IndexError(f"Index out of range: {i}")
        return self._uid_list[i]

    def uid_to_node(self, uid: str) -> Node:
        """Return Node dataclass for a uid."""
        return self._nodes[self.uid_to_index(uid)]

    @property
    def base_index(self) -> int:
        return self._base_index

    @property
    def dump_index(self) -> int:
        return self._dump_index

    @property
    def neighbor_precompute_seconds(self) -> float:
        """Seconds spent in neighbor precomputation (0.0 if disabled)."""
        return float(self._neighbor_precompute_s)

    def is_base(self, uid: str) -> bool:
        return self._kind_by_index[self.uid_to_index(uid)] == "base"

    def is_dump(self, uid: str) -> bool:
        return self._kind_by_index[self.uid_to_index(uid)] == "dump"

    def is_container(self, uid: str) -> bool:
        return self._kind_by_index[self.uid_to_index(uid)] == "container"

    def is_container_index(self, i: int) -> bool:
        return self._kind_by_index[i] == "container"

    def containers_uids(self) -> list[str]:
        """All container uids (excludes BASE and DUMP)."""
        return list(self._container_uids)

    def containers_indices(self) -> list[int]:
        """All container indices (excludes BASE and DUMP)."""
        return list(self._container_indices)

    def base_uid(self) -> str:
        """Uid of the base node (usually 'BASE')."""
        return self._base_uid

    def dump_uid(self) -> str:
        """Uid of the dump node (usually 'DUMP')."""
        return self._dump_uid

    def as_uid_list(self, indices: Sequence[int]) -> list[str]:
        """Convert a list of indices into uids."""
        return [self._uid_list[int(i)] for i in indices]

    def route_uids_to_idx(self, route: Sequence[str]) -> list[int]:
        """Convert a route expressed as uids into a route expressed as indices."""
        return [self.uid_to_index(u) for u in route]

    # ------------------------------
    # Travel times
    # ------------------------------
    @property
    def T(self) -> np.ndarray:
        """Dense travel time matrix in seconds (float32)."""
        return self._T

    def time_ij(self, i: int, j: int) -> float:
        """Travel time in seconds from index i to index j."""
        return float(self._T[i, j])

    def time_idx(self, i: int, j: int) -> float:
        """Alias of time_ij (explicit name used by some heuristics)."""
        return float(self._T[i, j])

    def time_uid(self, uid_a: str, uid_b: str) -> float:
        """Travel time in seconds from uid_a to uid_b."""
        return float(self._T[self.uid_to_index(uid_a), self.uid_to_index(uid_b)])

    def delta_insert_uid(self, a: str, b: str, x: str) -> float:
        """
        Travel-time delta of inserting x between a->b:

            delta = time(a,x) + time(x,b) - time(a,b)

        Useful for local search / insertion heuristics.
        """
        ia = self.uid_to_index(a)
        ib = self.uid_to_index(b)
        ix = self.uid_to_index(x)
        T = self._T
        return float(T[ia, ix] + T[ix, ib] - T[ia, ib])

    # ------------------------------
    # Helper utilities for heuristics
    # ------------------------------
    def k_nearest(
        self,
        uid: str,
        k: int,
        *,
        only_containers: bool = True,
        exclude: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """
        Return the k nearest nodes to `uid` by travel time.

        By default, returns only containers (excludes BASE/DUMP).
        You can exclude already-visited uids by passing `exclude`.

        Performance:
          - Uses precomputed row ordering if available.
          - Falls back to scanning otherwise.
        """
        if k <= 0:
            return []

        exclude = exclude or set()
        src_i = self.uid_to_index(uid)

        # Fast path: precomputed sorted indices for this row
        if self._row_sorted_idx is not None:
            res: list[tuple[str, float]] = []
            order = self._row_sorted_idx[src_i]
            for j in order:
                jj = int(j)
                if jj == src_i:
                    continue
                uid_j = self._uid_list[jj]
                if uid_j in exclude:
                    continue
                if only_containers and self._kind_by_index[jj] != "container":
                    continue
                res.append((uid_j, float(self._T[src_i, jj])))
                if len(res) >= k:
                    break
            return res

        # Fallback: scan all nodes
        candidates: list[tuple[str, float]] = []
        for j, uid_j in enumerate(self._uid_list):
            if j == src_i:
                continue
            if uid_j in exclude:
                continue
            if only_containers and self._kind_by_index[j] != "container":
                continue
            candidates.append((uid_j, float(self._T[src_i, j])))

        candidates.sort(key=lambda x: x[1])
        return candidates[:k]

    def travel_time_route_uids(self, route: Sequence[str]) -> float:
        """
        Travel time (seconds) of a route given as a sequence of uids.
        Example route: ["BASE", "c1", "c2", "DUMP", "BASE"]
        """
        if len(route) < 2:
            return 0.0
        idx = [self.uid_to_index(u) for u in route]
        return self.travel_time_route_idx(idx)

    def travel_time_route_idx(self, route_idx: Sequence[int]) -> float:
        """Travel time (seconds) of a route given as indices."""
        if len(route_idx) < 2:
            return 0.0
        total = 0.0
        T = self._T
        for a, b in zip(route_idx[:-1], route_idx[1:]):
            total += float(T[int(a), int(b)])
        return total

    def service_time_route_uids(self, route: Sequence[str]) -> float:
        """
        Service time (seconds) for a route sequence.
        Adds container service time when visiting containers, and dump service time when visiting DUMP.
        BASE has no service time.
        """
        if not route:
            return 0.0
        idx = [self.uid_to_index(u) for u in route]
        return self.service_time_route_idx(idx)

    def service_time_route_idx(self, route_idx: Sequence[int]) -> float:
        """Service time (seconds) for a route given as indices."""
        total = 0.0
        for i in route_idx:
            kind = self._kind_by_index[int(i)]
            if kind == "container":
                total += self.service_time_container_s
            elif kind == "dump":
                total += self.service_time_dump_s
        return float(total)

    def total_time_route_uids(self, route: Sequence[str]) -> float:
        """Travel time + service time for a route."""
        idx = [self.uid_to_index(u) for u in route]
        return self.total_time_route_idx(idx)

    def total_time_route_idx(self, route_idx: Sequence[int]) -> float:
        """Travel time + service time for a route (indices)."""
        return self.travel_time_route_idx(route_idx) + self.service_time_route_idx(route_idx)

    def route_is_closed(self, route: Sequence[str]) -> bool:
        """
        Check the required closure convention:
          - route starts at BASE
          - ends with DUMP, BASE (last stop before BASE is DUMP)
        """
        base = self._base_uid
        dump = self._dump_uid
        return len(route) >= 3 and route[0] == base and route[-2] == dump and route[-1] == base
