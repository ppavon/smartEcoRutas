from __future__ import annotations

"""
Export visual de rutas para inspección (NO afecta a la evaluación).

Salida principal:
  - KMZ (Google Earth) con rutas y puntos.
  - GPKG con capas geográficas auxiliares.

Importante para alumnos:
  - Si faltan assets NB11 requeridos, se lanza error grave (modo estricto).
  - Este módulo no cambia puntuaciones: solo sirve para visualizar resultados.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import zipfile
import html
import pickle

import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import linemerge

from framework.problem_instance import ProblemInstance


# Optional but strongly recommended for "routes on roads"
# (These are runtime deps; if missing, the exporter will fail loudly.)
import networkx as nx
import osmnx as ox


# ============================================================
# Google Earth KMZ (KML zipped)
# ============================================================

def _kml_color_abgr(hex_rgb: str, alpha: int = 220) -> str:
    """
    KML uses aabbggrr hex colors.
    hex_rgb: 'RRGGBB'
    alpha: 0-255
    """
    hex_rgb = hex_rgb.strip("#")
    rr = hex_rgb[0:2]
    gg = hex_rgb[2:4]
    bb = hex_rgb[4:6]
    aa = f"{alpha:02x}"
    return f"{aa}{bb}{gg}{rr}"


# ============================================================
# Road-based geometry helpers (using precomputed G_aug + uid_to_node)
# ============================================================

class GeoExportDataError(RuntimeError):
    """Raised when precomputed graph assets do not match the current problem/routes."""
    pass


class GeoExportStats:
    def __init__(self) -> None:
        self.segments_total = 0
        self.segments_directed_ok = 0
        self.segments_used_undirected = 0
        self.segments_used_straight = 0
        self.examples: List[str] = []

    def add_example(self, s: str, max_examples: int = 5) -> None:
        if len(self.examples) < max_examples:
            self.examples.append(s)


def _load_graph_assets(
    repo_root: Path,
    *,
    tag: str,
    assets_dirname: str = "nb11_outputs",
) -> Tuple[Any, Dict[str, Any], Path, Path]:
    """
    Loads:
      - Nb11Output_{tag}_G_aug.graphml
      - Nb11Output_{tag}_uid_to_node.pkl
    from {repo_root}/{assets_dirname}/
    """
    assets_dir = repo_root / assets_dirname
    if not assets_dir.exists():
        raise GeoExportDataError(
            f"No existe la carpeta de assets '{assets_dir}'. "
            f"Esperaba '{assets_dirname}' en el repo (ej: <repo>/{assets_dirname}/...)."
        )

    graph_path = assets_dir / f"Nb11Output_{tag}_G_aug.graphml"
    map_path = assets_dir / f"Nb11Output_{tag}_uid_to_node.pkl"

    if not graph_path.exists():
        raise GeoExportDataError(f"Falta el grafo: {graph_path}")
    if not map_path.exists():
        raise GeoExportDataError(f"Falta el mapping uid->node: {map_path}")

    try:
        G = ox.load_graphml(graph_path)
    except Exception as e:
        raise GeoExportDataError(f"No puedo cargar graphml '{graph_path}': {e}") from e

    try:
        uid_to_graphnode = pickle.loads(map_path.read_bytes())
    except Exception as e:
        raise GeoExportDataError(f"No puedo cargar pkl '{map_path}': {e}") from e

    if not isinstance(uid_to_graphnode, dict) or len(uid_to_graphnode) == 0:
        raise GeoExportDataError(
            f"El mapping '{map_path.name}' no parece un dict no-vacío. Tipo={type(uid_to_graphnode)} "
            f"size={getattr(uid_to_graphnode, '__len__', lambda: 'n/a')()}"
        )

    return G, uid_to_graphnode, graph_path, map_path


def _validate_graph_assets_against_problem(
    *,
    problem: ProblemInstance,
    routes: List[List[str]],
    uid_to_graphnode: Dict[str, Any],
    G: Any,
    tag: str,
    max_report: int = 20,
) -> None:
    """
    “Que me salte a la cara” si hay incoherencias:
      - uids en rutas que no existen en problem
      - uids en rutas que no están en uid_to_graphnode (except base/dump)
      - node_ids del mapping que no existen en el grafo
      - (opcional) contenedores del problem que faltan en mapping (solo aviso fuerte)
    """
    missing_in_problem = []
    missing_in_mapping = []
    bad_nodeids = []

    def _kind(uid: str) -> Optional[str]:
        try:
            return problem.uid_to_node(uid).kind
        except Exception:
            return None

    for ridx, route in enumerate(routes, start=1):
        if not route:
            raise GeoExportDataError(f"Ruta vacía en ridx={ridx}")

        for uid in route:
            try:
                n = problem.uid_to_node(uid)
            except Exception:
                missing_in_problem.append((ridx, uid))
                continue

            if n.kind == "container":
                if uid not in uid_to_graphnode:
                    missing_in_mapping.append((ridx, uid, n.kind))

    G_nodes_set = set(G.nodes)
    G_nodes_set_str = set(str(x) for x in G.nodes)

    used_uids = set(uid for route in routes for uid in route if uid is not None)
    for uid in used_uids:
        if uid in uid_to_graphnode:
            nid = uid_to_graphnode[uid]
            if (nid not in G_nodes_set) and (str(nid) not in G_nodes_set_str):
                bad_nodeids.append((uid, nid))

    msgs = []
    if missing_in_problem:
        sample = missing_in_problem[:max_report]
        msgs.append(
            "UIDs presentes en 'routes' pero NO existen en 'problem' (uid_to_node falla). "
            f"Ejemplos (ridx, uid): {sample}  (total={len(missing_in_problem)})"
        )
    if missing_in_mapping:
        sample = missing_in_mapping[:max_report]
        msgs.append(
            "UIDs de CONTENEDOR presentes en rutas pero NO están en uid_to_node.pkl. "
            f"Ejemplos (ridx, uid, kind): {sample}  (total={len(missing_in_mapping)})"
        )
    if bad_nodeids:
        sample = bad_nodeids[:max_report]
        msgs.append(
            "El uid_to_node.pkl referencia node_ids que NO existen en el grafo G_aug.graphml. "
            f"Ejemplos (uid, node_id): {sample}  (total={len(bad_nodeids)})"
        )

    problem_container_uids = [n.uid for n in problem.nodes if n.kind == "container"]
    not_covered = [uid for uid in problem_container_uids if uid not in uid_to_graphnode]
    if not_covered:
        ratio = len(not_covered) / max(1, len(problem_container_uids))
        if ratio > 0.05:
            sample = not_covered[:max_report]
            msgs.append(
                f"El mapping uid_to_node.pkl NO cubre una parte relevante de los contenedores del problem "
                f"(tag='{tag}'): faltan {len(not_covered)}/{len(problem_container_uids)} (~{ratio:.1%}). "
                f"Ejemplos: {sample}"
            )

    if msgs:
        raise GeoExportDataError("INCONSISTENCIA en assets precomputados vs problem/routes:\n- " + "\n- ".join(msgs))


def _graph_node_for_uid(
    *,
    uid: str,
    problem: ProblemInstance,
    uid_to_graphnode: Dict[str, Any],
    G: Any,
) -> Any:
    """
    Devuelve el node_id del grafo para ese uid.
    - Si uid está en mapping: lo usa
    - Si no (base/dump típicamente): cae a nearest_nodes por lon/lat
    """
    if uid in uid_to_graphnode:
        return uid_to_graphnode[uid]

    n = problem.uid_to_node(uid)
    try:
        return ox.distance.nearest_nodes(G, X=float(n.lon), Y=float(n.lat))
    except Exception as e:
        raise GeoExportDataError(
            f"No puedo encontrar nodo de grafo para uid='{uid}' (kind={n.kind}) ni por mapping ni por nearest_nodes: {e}"
        ) from e


def _route_geometry_from_graph(
    *,
    route: List[str],
    problem: ProblemInstance,
    G: Any,
    uid_to_graphnode: Dict[str, Any],
    weight: str = "length",
    global_stats: GeoExportStats | None = None,
) -> Tuple[LineString, Dict[str, int]]:
    """
    Construye una LineString siguiendo carretera, tramo a tramo entre stops consecutivos.
    Política de fallback:
      1) directed shortest_path en G
      2) si falla: undirected shortest_path (solo visualización)
      3) si falla: línea recta (lon/lat problema)

    Devuelve:
      - geometry
      - stats dict por ruta: {segments_total, directed_ok, used_undirected, used_straight}
    """
    if len(route) < 2:
        n = problem.uid_to_node(route[0])
        return LineString([(float(n.lon), float(n.lat)), (float(n.lon), float(n.lat))]), {
            "segments_total": 0,
            "directed_ok": 0,
            "used_undirected": 0,
            "used_straight": 0,
        }

    gnodes = [
        _graph_node_for_uid(uid=uid, problem=problem, uid_to_graphnode=uid_to_graphnode, G=G)
        for uid in route
    ]

    G_und = None
    per = {"segments_total": 0, "directed_ok": 0, "used_undirected": 0, "used_straight": 0}
    pieces: List[LineString] = []

    for i, (a, b) in enumerate(zip(gnodes[:-1], gnodes[1:])):
        if a == b:
            continue

        per["segments_total"] += 1
        if global_stats is not None:
            global_stats.segments_total += 1

        path = None

        # 1) DIRECTED
        try:
            path = nx.shortest_path(G, a, b, weight=weight)
            per["directed_ok"] += 1
            if global_stats is not None:
                global_stats.segments_directed_ok += 1
        except nx.NetworkXNoPath:
            path = None
        except Exception as e:
            raise GeoExportDataError(
                f"Error calculando shortest_path DIRECTED ({a}->{b}) weight='{weight}': {e}"
            ) from e

        # 2) UNDIRECTED (visualización)
        if path is None:
            if G_und is None:
                G_und = G.to_undirected(as_view=True)
            try:
                path = nx.shortest_path(G_und, a, b, weight=weight)
                per["used_undirected"] += 1
                if global_stats is not None:
                    global_stats.segments_used_undirected += 1
                    global_stats.add_example(f"UNDIRECTED tramo {i+1}: {a} -> {b}")
            except nx.NetworkXNoPath:
                path = None
            except Exception as e:
                raise GeoExportDataError(
                    f"Error calculando shortest_path UNDIRECTED ({a}->{b}) weight='{weight}': {e}"
                ) from e

        # 3) STRAIGHT fallback
        if path is None:
            ua, ub = route[i], route[i + 1]
            na, nb = problem.uid_to_node(ua), problem.uid_to_node(ub)
            pieces.append(LineString([(float(na.lon), float(na.lat)), (float(nb.lon), float(nb.lat))]))
            per["used_straight"] += 1
            if global_stats is not None:
                global_stats.segments_used_straight += 1
                global_stats.add_example(f"STRAIGHT tramo {i+1}: {ua} -> {ub}")
            continue

        # Convertir path a geometría (edges->geometry si existe)
        try:
            gdf_edges = ox.utils_graph.route_to_gdf(G, path, nodes=False)
        except Exception:
            gdf_edges = None

        if gdf_edges is not None and len(gdf_edges) > 0 and "geometry" in gdf_edges.columns:
            geom = linemerge(list(gdf_edges.geometry))
            if geom.geom_type == "MultiLineString":
                coords = []
                for part in geom.geoms:
                    coords.extend(list(part.coords))
                geom = LineString(coords)
            pieces.append(geom)
        else:
            coords = [(float(G.nodes[nid]["x"]), float(G.nodes[nid]["y"])) for nid in path]
            pieces.append(LineString(coords))

    merged = linemerge(pieces) if pieces else LineString()
    if merged.geom_type == "MultiLineString":
        coords = []
        for part in merged.geoms:
            coords.extend(list(part.coords))
        merged = LineString(coords)

    if merged.is_empty:
        coords = []
        for uid in route:
            n = problem.uid_to_node(uid)
            coords.append((float(n.lon), float(n.lat)))
        merged = LineString(coords)

    # --- Reproject geometry to EPSG:4326 if needed ---
    graph_crs = G.graph.get("crs", None)

    if graph_crs is not None:
        graph_crs_str = str(graph_crs).upper()
        if graph_crs_str != "EPSG:4326":
            try:
                gdf_tmp = gpd.GeoDataFrame(geometry=[merged], crs=graph_crs)
                gdf_tmp = gdf_tmp.to_crs("EPSG:4326")
                merged = gdf_tmp.geometry.iloc[0]
            except Exception as e:
                raise GeoExportDataError(
                    f"No puedo reproyectar geometría de {graph_crs} a EPSG:4326: {e}"
                )

    return merged, per

def _write_kmz_google_earth_roadbased(
    kmz_path: Path,
    *,
    problem: ProblemInstance,
    routes: List[List[str]],
    G: Any,
    uid_to_graphnode: Dict[str, Any],
    weight: str = "length",
    global_stats: GeoExportStats | None = None,
) -> None:
    """
    KMZ for Google Earth:
      - Folder per route (toggle visibility)
      - Nodes folder (BASE/DUMP highlighted; containers small)
      - Styles per route (distinct colors)
      - Route LineString follows roads (shortest paths on G_aug)
      - Si falla: undirected y/o recta (quedará reflejado en descripción)
    """
    palette = [
        "1f5ac8", "c83c3c", "3ca050", "a050b4", "f08c32",
        "46aac8", "c8b43c", "5a5a5a", "0078c8", "c80078",
        "00a078", "783c00", "787800", "003ca0", "a0003c",
        "3c00a0", "008c3c", "8c3c00", "3c8c00", "000000",
    ]

    def esc(s: str) -> str:
        return html.escape(str(s), quote=True)

    # Route styles
    style_blocks: List[str] = []
    for ridx in range(1, len(routes) + 1):
        rgb = palette[(ridx - 1) % len(palette)]
        line_color = _kml_color_abgr(rgb, alpha=230)
        style_blocks.append(f"""
    <Style id="route_{ridx:02d}">
      <LineStyle>
        <color>{line_color}</color>
        <width>3</width>
      </LineStyle>
    </Style>
""")

    # Node styles
    style_blocks.append("""
    <Style id="base_dump">
      <IconStyle>
        <scale>1.5</scale>
        <color>ff0000ff</color>
        <Icon><href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href></Icon>
      </IconStyle>
    </Style>
""")
    style_blocks.append("""
    <Style id="container">
      <IconStyle>
        <scale>0.6</scale>
        <color>ffaaaaaa</color>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
""")

    # Nodes folder
    nodes_pm: List[str] = []
    for n in problem.nodes:
        style = "#container" if n.kind == "container" else "#base_dump"
        name = n.uid
        desc = f"uid={esc(n.uid)}<br/>kind={esc(n.kind)}"
        nodes_pm.append(f"""
      <Placemark>
        <name>{esc(name)}</name>
        <description><![CDATA[{desc}]]></description>
        <styleUrl>{style}</styleUrl>
        <Point><coordinates>{float(n.lon)},{float(n.lat)},0</coordinates></Point>
      </Placemark>
""")

    # Routes folder
    routes_folders: List[str] = []
    for ridx, route in enumerate(routes, start=1):
        geom, per = _route_geometry_from_graph(
            route=route,
            problem=problem,
            G=G,
            uid_to_graphnode=uid_to_graphnode,
            weight=weight,
            global_stats=global_stats,
        )
        coords_kml = " ".join([f"{float(x)},{float(y)},0" for (x, y) in geom.coords])

        kinds: List[str] = []
        for uid in route:
            n = problem.uid_to_node(uid)
            kinds.append(n.kind)

        travel_time_s = float(problem.travel_time_route_uids(route))
        service_time_s = float(problem.service_time_route_uids(route))
        total_time_s = float(travel_time_s + service_time_s)

        travel_time_h = _sec_to_h(travel_time_s)
        service_time_h = _sec_to_h(service_time_s)
        total_time_h = _sec_to_h(total_time_s)
        num_stops = int(len(route))

        num_containers = int(sum(1 for k in kinds if k == "container"))
        dump_visits = int(sum(1 for k in kinds if k == "dump"))

        desc = (
            f"<b>Route {ridx:02d}</b><br/>"
            f"num_stops: {num_stops}<br/>"
            f"num_containers: {num_containers}<br/>"
            f"travel_time_s: {travel_time_s:.1f}<br/>"
            f"service_time_s: {service_time_s:.1f}<br/>"
            f"total_time_s: {total_time_s:.1f}<br/>"
            f"<br/><b>Road routing diagnostics</b><br/>"
            f"road_segments_total: {per['segments_total']}<br/>"
            f"road_used_undirected: {per['used_undirected']}<br/>"
            f"road_used_straight: {per['used_straight']}<br/>"
        )

        routes_folders.append(f"""
    <Folder>
      <name>Route {ridx:02d}</name>
      <open>0</open>
      <Placemark>
        <name>Route {ridx:02d}</name>
        <description><![CDATA[{desc}]]></description>
        <styleUrl>#route_{ridx:02d}</styleUrl>
        <LineString>
          <tessellate>1</tessellate>
          <coordinates>
            {coords_kml}
          </coordinates>
        </LineString>
      </Placemark>
    </Folder>
""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>SmartEcoRutas export</name>

  {''.join(style_blocks)}

  <Folder>
    <name>Nodes</name>
    <open>0</open>
    {''.join(nodes_pm)}
  </Folder>

  <Folder>
    <name>Routes</name>
    <open>1</open>
    {''.join(routes_folders)}
  </Folder>

</Document>
</kml>
"""

    kmz_path.parent.mkdir(parents=True, exist_ok=True)
    if kmz_path.exists():
        kmz_path.unlink()

    with zipfile.ZipFile(kmz_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)


def _write_kmz_google_earth_simple(
    kmz_path: Path,
    *,
    problem: ProblemInstance,
    routes: List[List[str]],
) -> None:
    """
    KMZ simple (sin assets de grafo): líneas rectas entre paradas y
    puntos de contenedores por ruta.
    """
    palette = [
        "1f5ac8", "c83c3c", "3ca050", "a050b4", "f08c32",
        "46aac8", "c8b43c", "5a5a5a", "0078c8", "c80078",
    ]

    def esc(s: str) -> str:
        return html.escape(str(s), quote=True)

    style_blocks: List[str] = []
    for ridx in range(1, len(routes) + 1):
        rgb = palette[(ridx - 1) % len(palette)]
        line_color = _kml_color_abgr(rgb, alpha=230)
        style_blocks.append(f"""
    <Style id="route_{ridx:02d}">
      <LineStyle><color>{line_color}</color><width>3</width></LineStyle>
      <IconStyle>
        <scale>0.7</scale>
        <color>{line_color}</color>
        <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>
      </IconStyle>
    </Style>
""")

    route_folders: List[str] = []
    for ridx, route in enumerate(routes, start=1):
        coords = []
        for uid in route:
            n = problem.uid_to_node(uid)
            coords.append((float(n.lon), float(n.lat)))
        coords_kml = " ".join([f"{lon},{lat},0" for lon, lat in coords])

        travel_s = float(problem.travel_time_route_uids(route))
        total_s = float(problem.total_time_route_uids(route))
        n_cont = sum(1 for uid in route if problem.is_container(uid))
        n_dump = sum(1 for uid in route if uid == problem.dump_uid())

        cont_points = []
        for uid in route:
            n = problem.uid_to_node(uid)
            if n.kind != "container":
                continue
            cont_points.append(f"""
      <Placemark>
        <name>{esc(uid)}</name>
        <styleUrl>#route_{ridx:02d}</styleUrl>
        <Point><coordinates>{float(n.lon)},{float(n.lat)},0</coordinates></Point>
      </Placemark>
""")

        desc = (
            f"<b>Ruta {ridx:02d}</b><br/>"
            f"contenedores: {n_cont}<br/>"
            f"visitas_dump: {n_dump}<br/>"
            f"travel_h: {travel_s/3600.0:.2f}<br/>"
            f"total_h: {total_s/3600.0:.2f}<br/>"
            f"modo_geometria: simple_lineal (sin grafo)<br/>"
        )
        route_folders.append(f"""
    <Folder>
      <name>Ruta {ridx:02d}</name>
      <open>0</open>
      <Placemark>
        <name>Ruta {ridx:02d}</name>
        <description><![CDATA[{desc}]]></description>
        <styleUrl>#route_{ridx:02d}</styleUrl>
        <LineString><tessellate>1</tessellate><coordinates>{coords_kml}</coordinates></LineString>
      </Placemark>
      <Folder>
        <name>Contenedores Ruta {ridx:02d}</name>
        {''.join(cont_points)}
      </Folder>
    </Folder>
""")

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>SmartEcoRutas KMZ (simple)</name>
  {''.join(style_blocks)}
  <Folder>
    <name>Rutas</name>
    <open>1</open>
    {''.join(route_folders)}
  </Folder>
</Document>
</kml>
"""
    kmz_path.parent.mkdir(parents=True, exist_ok=True)
    if kmz_path.exists():
        kmz_path.unlink()
    with zipfile.ZipFile(kmz_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml)


# ============================================================
# Main export
# ============================================================

def _sec_to_h(sec: float) -> float:
    return float(sec) / 3600.0

def export_for_qgis(
    problem: ProblemInstance,
    routes: List[List[str]],
    out_dir: str | Path,
    *,
    nb11_tag: str | None = None,
    repo_root: str | Path = ".",
    nb11_outputs_dir: str = "nb11_outputs",
    road_weight: str = "length",
    gpkg_name: str = "export.gpkg",
    include_visits_layer: bool = True,
    kmz_name: str = "export.kmz",
) -> dict:
    """
    Exports:
      - export.gpkg with layers: nodes_base_dump, nodes_containers, routes (+ visits optional)
      - export.kmz for Google Earth (folders per route + nodes)

    Routes geometry tries to follow OSM roads using precomputed:
      - nb11_outputs/Nb11Output_{nb11_tag}_G_aug.graphml
      - nb11_outputs/Nb11Output_{nb11_tag}_uid_to_node.pkl

    If a directed road path is missing, exporter falls back to:
      - undirected (visualization only)
      - straight line (final fallback)

    IMPORTANT:
      - This does NOT affect evaluation; only visualization.
      - If warnings appear, it's usually due to OSM connectivity/attributes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(repo_root)

    # Auto-detect nb11_tag if not provided
    if nb11_tag is None:
        candidates = []
        for attr in ("name", "instance_name", "instance", "instance_id", "id"):
            if hasattr(problem, attr):
                val = getattr(problem, attr)
                if isinstance(val, str) and val.strip():
                    candidates.append(val.strip())

        od = str(out_dir)
        if od:
            candidates.append(Path(od).name)

        repo_root_path = Path(repo_root)
        assets_dir = repo_root_path / nb11_outputs_dir
        found = None
        for cand in candidates:
            p = assets_dir / f"Nb11Output_{cand}_G_aug.graphml"
            if p.exists():
                found = cand
                break

        if found is None:
            raise GeoExportDataError(
                "No se pudo inferir nb11_tag automáticamente.\n"
                "Pasa nb11_tag='LATERAL_CARTON' (o el tag correspondiente) al llamar export_for_qgis,\n"
                f"o asegúrate de que existan ficheros en {assets_dir} con patrón Nb11Output_<TAG>_G_aug.graphml.\n"
                f"Candidatos probados: {candidates}"
            )
        nb11_tag = found

    # -----------------------
    # Load assets & validate (estricto: si falla, se aborta)
    # -----------------------
    assets_ok = True
    assets_error: Optional[str] = None
    G = None
    uid_to_graphnode: Dict[str, Any] = {}
    graph_path: Optional[Path] = None
    map_path: Optional[Path] = None
    try:
        G, uid_to_graphnode, graph_path, map_path = _load_graph_assets(
            repo_root, tag=nb11_tag, assets_dirname=nb11_outputs_dir
        )
        _validate_graph_assets_against_problem(
            problem=problem,
            routes=routes,
            uid_to_graphnode=uid_to_graphnode,
            G=G,
            tag=nb11_tag,
        )
    except Exception as e:
        assets_ok = False
        assets_error = repr(e)
        raise GeoExportDataError(
            f"Fallo cargando/validando assets NB11 para export visual. Detalle: {assets_error}"
        ) from e

    # -----------------------
    # output files
    # -----------------------
    gpkg_path = out_dir / gpkg_name
    kmz_path = out_dir / kmz_name

    # overwrite
    if gpkg_path.exists():
        gpkg_path.unlink()
    if kmz_path.exists():
        kmz_path.unlink()

    # -----------------------
    # nodes split layers
    # -----------------------
    base_dump_rows, base_dump_geoms = [], []
    cont_rows, cont_geoms = [], []

    for n in problem.nodes:
        row = {"uid": n.uid, "kind": n.kind, "lon": float(n.lon), "lat": float(n.lat)}
        geom = Point(float(n.lon), float(n.lat))
        if n.kind in ("base", "dump"):
            base_dump_rows.append(row)
            base_dump_geoms.append(geom)
        else:
            cont_rows.append(row)
            cont_geoms.append(geom)

    gdf_base_dump = gpd.GeoDataFrame(base_dump_rows, geometry=base_dump_geoms, crs="EPSG:4326")
    gdf_cont = gpd.GeoDataFrame(cont_rows, geometry=cont_geoms, crs="EPSG:4326")

    gdf_base_dump.to_file(gpkg_path, layer="nodes_base_dump", driver="GPKG")
    gdf_cont.to_file(gpkg_path, layer="nodes_containers", driver="GPKG")

    # -----------------------
    # routes layer (all routes) -- ROAD-BASED GEOMETRY (with fallbacks)
    # -----------------------
    geo_stats = GeoExportStats()

    route_rows: List[Dict[str, Any]] = []
    route_geoms: List[LineString] = []

    for ridx, route in enumerate(routes, start=1):
        if not route:
            raise GeoExportDataError(f"Ruta vacía en ridx={ridx}")

        kinds = []
        for uid in route:
            n = problem.uid_to_node(uid)
            kinds.append(n.kind)

        travel_time_s = float(problem.travel_time_route_uids(route))
        service_time_s = float(problem.service_time_route_uids(route))
        total_time_s = float(travel_time_s + service_time_s)

        travel_time_h = _sec_to_h(travel_time_s)
        service_time_h = _sec_to_h(service_time_s)
        total_time_h = _sec_to_h(total_time_s)

        num_containers = int(sum(1 for k in kinds if k == "container"))
        dump_visits = int(sum(1 for k in kinds if k == "dump"))

        if assets_ok:
            geom, per = _route_geometry_from_graph(
                route=route,
                problem=problem,
                G=G,
                uid_to_graphnode=uid_to_graphnode,
                weight=road_weight,
                global_stats=geo_stats,
            )
        else:
            route_coords = [(float(problem.uid_to_node(u).lon), float(problem.uid_to_node(u).lat)) for u in route]
            geom = LineString(route_coords)
            per = {
                "segments_total": max(0, len(route) - 1),
                "directed_ok": 0,
                "used_undirected": 0,
                "used_straight": max(0, len(route) - 1),
            }

        route_rows.append(
            {
                "route_id": ridx,
                "num_stops": int(len(route)),
                "num_containers": int(num_containers),
                "dump_visits": dump_visits,
                "travel_time_h": round(travel_time_h, 3),
                "service_time_h": round(service_time_h, 3),
                "total_time_h": round(total_time_h, 3),
                "is_closed": bool(problem.route_is_closed(route)),
                "label": f"R{ridx:02d} | cont={num_containers} | dump={dump_visits} | t={total_time_h:.2f}h",

                # Diagnostics for QGIS filtering / marking
                "road_segments_total": int(per["segments_total"]),
                "road_directed_ok": int(per["directed_ok"]),
                "road_used_undirected": int(per["used_undirected"]),
                "road_used_straight": int(per["used_straight"]),
                "road_ok": bool(per["used_straight"] == 0),
            }
        )
        route_geoms.append(geom)

    gdf_routes = gpd.GeoDataFrame(route_rows, geometry=route_geoms, crs="EPSG:4326")
    gdf_routes.to_file(gpkg_path, layer="routes", driver="GPKG")

    # -----------------------
    # visits (optional)
    # -----------------------
    layers = ["nodes_base_dump", "nodes_containers", "routes"]
    if include_visits_layer:
        visit_rows, visit_geoms = [], []
        for ridx, route in enumerate(routes, start=1):
            for stop_idx, uid in enumerate(route, start=1):
                n = problem.uid_to_node(uid)
                visit_rows.append(
                    {"route_id": ridx, "stop_idx": int(stop_idx), "uid": n.uid, "kind": n.kind}
                )
                visit_geoms.append(Point(float(n.lon), float(n.lat)))
        gdf_visits = gpd.GeoDataFrame(visit_rows, geometry=visit_geoms, crs="EPSG:4326")
        gdf_visits.to_file(gpkg_path, layer="visits", driver="GPKG")
        layers.append("visits")

    # -----------------------
    # write KMZ (Google Earth) -- ROAD-BASED (with fallbacks)
    # -----------------------
    if assets_ok:
        _write_kmz_google_earth_roadbased(
            kmz_path,
            problem=problem,
            routes=routes,
            G=G,
            uid_to_graphnode=uid_to_graphnode,
            weight=road_weight,
            global_stats=geo_stats,
        )
    else:
        _write_kmz_google_earth_simple(
            kmz_path,
            problem=problem,
            routes=routes,
        )

    # -----------------------
    # console diagnostics (IMPORTANT: not student fault)
    # -----------------------
    if assets_ok and (geo_stats.segments_used_undirected > 0 or geo_stats.segments_used_straight > 0):
        print("\n[GeoExport][WARN] Algunas conexiones entre paradas NO pudieron rutearse en modo 'directed' usando OSM.")
        print("               Esto NO es un problema del algoritmo del estudiante. Es un detalle de conectividad/atributos del grafo OSM.")
        print(f"               Tramos total: {geo_stats.segments_total}")
        print(f"               Directed OK : {geo_stats.segments_directed_ok}")
        print(f"               Undirected  : {geo_stats.segments_used_undirected}  (visualización)")
        print(f"               Recta       : {geo_stats.segments_used_straight}  (fallback final)")
        if geo_stats.examples:
            print("               Ejemplos:")
            for s in geo_stats.examples[:5]:
                print("                 -", s)
        print("               Puedes IGNORAR este warning. La evaluación del reto NO depende de este export.\n")
    elif assets_ok:
        print(f"[GeoExport][OK] Geometría por carretera completada: {geo_stats.segments_total} tramos (100% directed).")
    else:
        print("[GeoExport][OK] KMZ generado en modo fallback simple (líneas rectas entre paradas).")

    return {
        "gpkg": gpkg_path.name,
        "kmz": kmz_path.name,
        "layers": layers,
        "geometry_mode": "road_based" if assets_ok else "simple_fallback",
        "assets_warning": assets_error,
        "road_fallback": {
            "segments_total": geo_stats.segments_total,
            "directed_ok": geo_stats.segments_directed_ok,
            "used_undirected": geo_stats.segments_used_undirected,
            "used_straight": geo_stats.segments_used_straight,
            "examples": geo_stats.examples[:5],
            "note": "Esto NO afecta a la evaluación. Es solo export/visualización; depende de conectividad OSM.",
        },
        "note": (
            "Google Earth: abre export.kmz. "
            + (
                f"Assets usados: {graph_path.name}, {map_path.name} (tag={nb11_tag}, weight={road_weight})."
                if assets_ok and graph_path is not None and map_path is not None
                else "Sin assets de grafo: se usó geometría lineal simple."
            )
        ),
    }
