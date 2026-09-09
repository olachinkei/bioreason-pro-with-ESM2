"""GO ontology helpers (AUTHORED). PROTECTED.

Load information-accretion weights (IA.txt) and a GO ancestor map (go-basic.obo) for the RL
reward's F_max proxy. The AUTHORITATIVE metric (eval.py) delegates propagation to `cafaeval`;
this ancestor map is only for the fast per-rollout reward proxy (True-Path-Rule expansion so the
proxy correlates better with cafaeval's propagated weighted F_max — calibrate in Stage 0).
"""

from __future__ import annotations

from pathlib import Path

# Propagation relations per the CAFA / True-Path Rule (regulates edges are NOT used — they can cycle).
GO_PROPAGATION_RELATIONS = ("is_a", "part_of")


def load_ia_weights(path: str | Path) -> dict[str, float]:
    """Parse IA.txt lines 'GO:xxxxxxx<TAB or space><ia_float>' into {term: weight}."""
    ia: dict[str, float] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) >= 2:
            try:
                ia[parts[0]] = float(parts[1])
            except ValueError:
                continue
    return ia


def load_go_ancestors(obo_path: str | Path,
                      relations: tuple[str, ...] = GO_PROPAGATION_RELATIONS) -> dict[str, set[str]]:
    """Return {go_term: set(all ancestors)} from go-basic.obo, following only `relations`.

    obonet yields a MultiDiGraph with edges child -> parent keyed by relation type. We build a
    filtered DiGraph (is_a/part_of only), topologically sort it (GO is a DAG under these relations),
    and fill ancestors in reverse-topo order (parents before children) so each node unions its
    parents' already-computed ancestors — linear, no per-node graph traversal.
    """
    import networkx as nx
    import obonet

    g = obonet.read_obo(obo_path)
    fg = nx.DiGraph()
    fg.add_nodes_from(g.nodes())
    if g.is_multigraph():
        for u, v, key in g.edges(keys=True):
            if key in relations:
                fg.add_edge(u, v)
    else:  # some obonet versions return a plain DiGraph
        fg.add_edges_from(g.edges())

    ancestors: dict[str, set[str]] = {n: set() for n in fg.nodes()}
    for node in reversed(list(nx.topological_sort(fg))):
        for parent in fg.successors(node):          # edge child -> parent
            ancestors[node].add(parent)
            ancestors[node] |= ancestors[parent]
    return ancestors


NAMESPACE_TO_ASPECT = {
    "molecular_function": "MF",
    "biological_process": "BP",
    "cellular_component": "CC",
}


def load_go_aspects(obo_path: str | Path) -> dict[str, str]:
    """Return {go_term: 'MF'|'BP'|'CC'} from go-basic.obo.

    The RL reward scores a flat set of predicted ids, but the evaluation metric is a mean over the
    three aspects, so an aspect-aware reward needs to know which aspect each predicted id belongs
    to. GO's aspects are disjoint subtrees, so every term has exactly one.
    """
    import obonet

    graph = obonet.read_obo(obo_path)
    aspects: dict[str, str] = {}
    for term, data in graph.nodes(data=True):
        aspect = NAMESPACE_TO_ASPECT.get(data.get("namespace", ""))
        if aspect:
            aspects[term] = aspect
    return aspects


def load_go_names(obo_path: str | Path) -> dict[str, str]:
    """Return {go_term: human-readable name} from go-basic.obo.

    Needed to check whether a reasoning trace actually *justifies* the terms it predicts: a reader
    from drug discovery reads "protein homodimerization activity", not `GO:0042803`, so a trace
    that never names its terms in words is not inspectable even when it is correct.
    """
    import obonet

    graph = obonet.read_obo(obo_path)
    return {term: data["name"] for term, data in graph.nodes(data=True) if data.get("name")}
