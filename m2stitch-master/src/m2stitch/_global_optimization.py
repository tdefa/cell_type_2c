import networkx as nx
import numpy as np
import pandas as pd

from ._typing_utils import Int


def compute_maximum_spanning_tree(grid: pd.DataFrame) -> nx.Graph:
    """Compute the maximum spanning tree for grid position determination.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position,
        with columns "{top|left}_{x|y|ncc|valid3}"

    Returns
    -------
    tree : nx.Graph
        the result spanning tree
    """
    connection_graph = nx.Graph()
    for i, g in grid.iterrows():
        for direction in ["top", "left"]:
            if not pd.isna(g[direction]):
                weight = g[f"{direction}_ncc"]
                if g[f"{direction}_valid3"]:
                    weight = weight + 10
                connection_graph.add_edge(
                    i,
                    g[direction],
                    weight=weight,
                    direction=direction,
                    f=i,
                    t=g[direction],
                    x=g[f"{direction}_x"],
                    y=g[f"{direction}_y"],
                )
    return nx.maximum_spanning_tree(connection_graph)


def compute_final_position(
    grid: pd.DataFrame, tree: nx.Graph, source_index: Int = 0
) -> pd.DataFrame:
    """Compute the final tile positions by the computed maximum spanning tree.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position
    tree : nx.Graph
        the maximum spanning tree
    source_index : Int, optional
        the source position of the spanning tree, by default 0

    Returns
    -------
    grid : pd.DataFrame
        the result dataframe for the grid position, with columns "{x|y}_pos"
    """
    grid.loc[source_index, "x_pos"] = 0
    grid.loc[source_index, "y_pos"] = 0

    nodes = [source_index]
    walked_nodes = []
    while len(nodes) > 0:
        node = nodes.pop()
        walked_nodes.append(node)
        for adj, props in tree.adj[node].items():
            if not adj in walked_nodes:
                assert (props["f"] == node) & (props["t"] == adj) or (
                    props["t"] == node
                ) & (props["f"] == adj)
                nodes.append(adj)
                x_pos = grid.loc[node, "x_pos"]
                y_pos = grid.loc[node, "y_pos"]

                if node == props["t"]:
                    grid.loc[adj, "x_pos"] = x_pos + props["x"]
                    grid.loc[adj, "y_pos"] = y_pos + props["y"]
                else:
                    grid.loc[adj, "x_pos"] = x_pos - props["x"]
                    grid.loc[adj, "y_pos"] = y_pos - props["y"]
    assert not any(pd.isna(grid["x_pos"]))
    assert not any(pd.isna(grid["y_pos"]))
    grid["x_pos"] = grid["x_pos"] - grid["x_pos"].min()
    grid["y_pos"] = grid["y_pos"] - grid["y_pos"].min()
    grid["x_pos"] = grid["x_pos"].astype(np.int32)
    grid["y_pos"] = grid["y_pos"].astype(np.int32)

    return grid
