import pytest
from luxonis_train.utils.general import infer_upscale_factor, is_acyclic, traverse_graph


def test_infer_upscale_factor():
    # Test with integer upscale factor
    upscale_factor = infer_upscale_factor(128, 256)
    assert upscale_factor == 1

    upscale_factor = infer_upscale_factor(128, 1024)
    assert upscale_factor == 3

    # Test with non-integer upscale factor, strict=False
    upscale_factor = infer_upscale_factor(100, 200, strict=False)
    assert upscale_factor == 1

    # Test with non-integer upscale factor, strict=True
    with pytest.raises(ValueError):
        infer_upscale_factor(100, 201, strict=True)


def test_is_acyclic():
    # Acyclic Graph
    graph_acyclic = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
    assert is_acyclic(graph_acyclic)

    # Complex Acyclic Graph
    graph_complex = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": ["G"],
        "E": ["G", "H"],
        "F": ["H"],
        "G": ["I"],
        "H": ["I"],
        "I": [],
    }
    assert is_acyclic(graph_complex)

    # Cyclic Graph
    graph_cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
    assert not is_acyclic(graph_cyclic)

    # Empty Graph
    graph_empty = {}
    assert is_acyclic(graph_empty)

    # Single-Node Graph
    graph_single_node = {"A": []}
    assert is_acyclic(graph_single_node)


def test_traverse_graph():
    # Properly Formed Directed Acyclic Graph
    graph = {"A": [], "B": ["A"], "C": ["B"]}
    nodes = {"A": 1, "B": 2, "C": 3}
    traversal_order = [name for name, _, _, _ in traverse_graph(graph, nodes)]
    assert traversal_order == ["A", "B", "C"]

    # Graph with a Cycle
    graph_cyclic = {"A": ["B"], "B": ["C"], "C": ["A"]}
    with pytest.raises(RuntimeError):
        list(traverse_graph(graph_cyclic, nodes))

    # Empty Graph
    graph_empty = {}
    assert list(traverse_graph(graph_empty, {})) == []

    # Graph with Isolated Nodes
    graph_isolated = {"A": [], "B": [], "C": []}
    traversal_isolated = [
        name for name, _, _, _ in traverse_graph(graph_isolated, nodes)
    ]
    assert set(traversal_isolated) == {"A", "B", "C"}

    # Complex Directed Graph
    graph_complex = {
        "A": ["B", "C"],
        "B": ["D", "E"],
        "C": ["F"],
        "D": ["G"],
        "E": ["G", "H"],
        "F": ["H"],
        "G": ["I"],
        "H": ["I"],
        "I": [],
    }
    nodes_complex = {node: node for node in graph_complex}

    traversal_order = [
        name for name, _, _, _ in traverse_graph(graph_complex, nodes_complex)
    ]

    # Validate the traversal order
    # Each node should appear after all its dependencies have been processed
    for node, dependencies in graph_complex.items():
        node_index = traversal_order.index(node)
        for dependency in dependencies:
            assert traversal_order.index(dependency) < node_index
