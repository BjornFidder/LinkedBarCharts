#%%

import igraph as ig
import numpy as np
import networkx as nx

def edge_density(graph):
    n = graph.vcount()
    m = graph.ecount()
    return 2 * m / (n * (n-1))

def teenage_friends(weight_range=(1,10)):

    path = "C:/Users/bjorn/OneDrive - Universiteit Utrecht/UNI/Master/Jaar 2/Semester 2/Experimentation Project/Networks/TeenageFriends/"
    files = [path+"s50-network1.dat", path+"s50-network2.dat", path+"s50-network3.dat"]
    
    graphs = []
    
    for file in files:
        
        graph = ig.Graph.Read_Adjacency(file)

        graph.to_undirected(mode='collapse')

        for edge in graph.es:
            edge['weight'] = np.random.randint(*weight_range)

        graph["vs_pos"] = np.array(graph.vs)
        np.random.shuffle(graph["vs_pos"])

        for (i, v) in enumerate(graph["vs_pos"]):
            v["pos"] = i

        graphs.append(graph)

    return graphs

def roll_call_votes(weight_threshold=0.8, weight_range=(1, 10), edge_density=None, mat_max=-2):

    path = "C:/Users/bjorn/OneDrive - Universiteit Utrecht/UNI/Master/Jaar 2/Semester 2/Experimentation Project/Networks/RollCallVotes/"
    files = [path+f"mat{i}.txt" for i in range(1, 11)][:mat_max+1]
    
    graphs = []
    
    for file in files:
        
        matrix = np.loadtxt(file)
        graph = ig.Graph.Weighted_Adjacency(matrix)

        graph.to_undirected(mode='collapse', combine_edges = 'mean')

        if edge_density is not None:
            weights = sorted([edge['weight'] for edge in graph.es])
            m = graph.ecount()
            n = graph.vcount()
            remove_count = round(m - edge_density * n * (n-1) / 2)
            remove_ids = range(remove_count)

        else:
            remove_ids = []
            for edge in graph.es:
                if edge['weight'] < weight_threshold:
                    remove_ids.append(edge.index)

        graph.delete_edges(remove_ids)

        for edge in graph.es:
            weight_min, weight_max = weight_range
            edge['weight'] = weight_min + round((edge['weight'] - weight_threshold) / (1 - weight_threshold) * (weight_max - weight_min))


        graph["vs_pos"] = np.array(graph.vs)
        np.random.shuffle(graph["vs_pos"])

        for (i, v) in enumerate(graph["vs_pos"]):
            v["pos"] = i

        graphs.append(graph)

    return graphs


def movies(weight_range=(1,10), load_range=(1, 915), n_range=(1, 200)):

    path = "C:/Users/bjorn/OneDrive - Universiteit Utrecht/UNI/Master/Jaar 2/Semester 2/Experimentation Project/Networks/Movies/"
    files = [path+f"{i}.gexf" for i in range(*load_range)]
    
    graphs = []
    
    for file in files:
        try:
            network = nx.read_gexf(file)
        except:
            continue
        if network.number_of_nodes() < n_range[0] or network.number_of_nodes() > n_range[1]:
            continue
        graph = ig.Graph.from_networkx(network)

        max_weight = max([edge["weight"] for edge in graph.es])
        for edge in graph.es:
            edge["weight"] = round(10 / max_weight * edge["weight"])

        graph["vs_pos"] = np.array(graph.vs)
        np.random.shuffle(graph["vs_pos"])

        for (i, v) in enumerate(graph["vs_pos"]):
            v["pos"] = i

        graphs.append(graph)

    return graphs
