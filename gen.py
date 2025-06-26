#%%
import igraph as ig
import numpy as np

def random_graph(n, p, weight_range=(1,10)):
    
    graph = ig.Graph.Erdos_Renyi(n, p)

    for edge in graph.es:
        edge['weight'] = np.random.randint(*weight_range)

    for (i, vertex) in enumerate(graph.vs):
        vertex["pos"] = i

    graph["vs_pos"] = np.array(graph.vs)

    return graph

def distance(v1, v2):
    return np.sqrt(np.sum((v1["coordinates"] - v2["coordinates"])**2))

def random_geometric_graph(n, r, dim=2, weight_range=(1,10)):

    graph = ig.Graph()

    for i in range(n):
        v = graph.add_vertex()
        v["coordinates"] = np.random.rand(dim)
        graph.add_edges([(v, w) for w in graph.vs() if (w!=v and distance(v, w) < r*np.sqrt(dim))])
    
    for edge in graph.es:
        edge['weight'] = np.random.randint(*weight_range)
    
    for (i, vertex) in enumerate(graph.vs):
        vertex["pos"] = i

    graph["vs_pos"] = np.array(graph.vs)

    return graph




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    dim = 2
    graph = random_geometric_graph(50, 0.4, dim=dim)
    fig,ax = plt.subplots()

    layout = ig.Layout([v["coordinates"][:2] for v in graph.vs()])

    ig.plot(graph, target=ax, layout=layout)
    plt.show()
    