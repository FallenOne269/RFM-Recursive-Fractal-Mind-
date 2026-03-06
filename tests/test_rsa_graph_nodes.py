import numpy as np

from rfai.fractal_graph import FractalGraph
from rfai.smart_fractal_node import SmartFractalNode
from rfai.dfe import DynamicFractalEncoder
from rfai.rsa import RecursiveStructuralAdapter, RSAConfig
from rfai.adaptation_signal import AdaptationSignal


def test_graph_depth_and_nodes():
    graph = FractalGraph(max_depth=1)
    dfe = DynamicFractalEncoder()
    fim = dfe.encode("root", np.ones(2), context={})
    node = SmartFractalNode(node_id="root", fim_id=fim.fim_id)
    graph.add_node(node)
    child = SmartFractalNode(node_id="child", fim_id=fim.fim_id)
    try:
        graph.add_node(child, parent_id="root")
    except ValueError:
        pass
    assert graph.depth("root") == 0


def test_rsa_bifurcation_and_consolidation():
    dfe = DynamicFractalEncoder()
    fim = dfe.encode("root", np.ones(2), context={})
    graph = FractalGraph(max_depth=3)
    node = SmartFractalNode(node_id="root", fim_id=fim.fim_id)
    graph.add_node(node)
    rsa = RecursiveStructuralAdapter(dfe, graph, RSAConfig(max_depth=3, max_nodes=3))
    sig = AdaptationSignal(delta=0.6, reason="grow")
    rsa.apply_signal("root", sig)
    assert len(graph.nodes) >= 1
    # consolidation path
    sig2 = AdaptationSignal(delta=-0.1, reason="stabilize")
    rsa.apply_signal("root", sig2)
    assert "root" in graph.nodes


def test_propagation():
    graph = FractalGraph(max_depth=3)
    dfe = DynamicFractalEncoder()
    fim = dfe.encode("root", np.ones(2), context={})
    root = SmartFractalNode(node_id="root", fim_id=fim.fim_id)
    graph.add_node(root)
    child = SmartFractalNode(node_id="child", fim_id=fim.fim_id)
    graph.add_node(child, parent_id="root")
    visited = graph.propagate(AdaptationSignal(delta=0.1, reason="test"), "root")
    assert ("root",) == visited[0][:1]

