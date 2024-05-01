
import numpy as np
import networkx as nx


def random_walk_subgraph(graph, start_node, walk_length, num_neighbors):
    subgraph = nx.Graph()
    visited = set()
    current_node = start_node

    for _ in range(walk_length):
        neighbors = list(graph.neighbors(current_node))
        if len(neighbors) == 0:
            break
        next_node = np.random.choice(neighbors)
        subgraph.add_edge(current_node, next_node)
        visited.add(current_node)
        visited.add(next_node)
        current_node = next_node
        if len(visited) >= num_neighbors:
            break

    return subgraph, nx.adjacency_matrix(graph.subgraph(visited))


edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6), (5, 7), (6, 7)]
graph = nx.Graph()
graph.add_edges_from(edges)
start_node = np.random.choice(list(graph.nodes()))
walk_length = 1500
num_neighbors = 1500
subgraph, adjacency_matrix = random_walk_subgraph(graph, start_node, walk_length, num_neighbors)

print("subgraph nodes：", subgraph.nodes())
print("subgraph edges：", subgraph.edges())
print("subgraph matrix：\n", adjacency_matrix.todense())
