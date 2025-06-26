import numpy as np
import bars
from blocks import inter_vertices


def ELL(edge, graph):
    #Compute an estimate for the (horizontal+vertical) Link Length of an edge
    inter_heights = [v["height"] for v in inter_vertices(edge, graph)]
    return np.sum([inter_heights])

def vertex_ELL(v):
    #Compute estimate for the link lengths of all edges passing over vertex v
    return (np.sqrt(v["height"])+1) * v["ecount"]

def compute_heights(graph):
    #(pre)Compute all vertex heights
    for v in graph.vs():
        v["height"] = np.sum([e["weight"] for e in v.incident()])

def compute_counts(graph):
     #Compute the number of edges passing over each vertex
     for v in graph.vs():
          v["ecount"] = 0 #number of edges passing over v
     for edge in graph.es():
          for v in inter_vertices(edge, graph):
               v["ecount"] += 1
    
def total_ELL(graph):
    compute_heights(graph)
    compute_counts(graph)
    return np.sum([vertex_ELL(v) for v in graph.vs()])

def swap(graph, u, v):
     swap_counts(u, v) 
     u["pos"], v["pos"] = v["pos"], u["pos"]
     graph["vs_pos"][u["pos"]], graph["vs_pos"][v["pos"]] = graph["vs_pos"][v["pos"]], graph["vs_pos"][u["pos"]]

def swap_counts(u, v):
     #Update the counts of vertices u, v if they are swapped
     for w in v.neighbors():
        if w["pos"] < u["pos"]:
            u["ecount"] -= 1
        if w["pos"] > v["pos"]:
            u["ecount"] += 1
     for w in u.neighbors():
        if w["pos"] > v["pos"]:
            v["ecount"] -= 1
        if w["pos"] < u["pos"]:
            v["ecount"] += 1
     
def delta_swap(u, v):
    #Compute the difference in ELL when swapping vertices u, v
    swap_counts(u, v)
    delta = vertex_ELL(u) + vertex_ELL(v)
    u["pos"], v["pos"] = v["pos"], u["pos"]
    swap_counts(v, u)
    delta -= vertex_ELL(u) + vertex_ELL(v)
    return delta

def adj_2opt(graph, maxiters=100):
    #Iterate 2-OPT, only swapping the positions of adjacent vertices  
    graph["ELL"] = total_ELL(graph)
    local_opt = False
    i = 0
    while not local_opt and i < maxiters:
        i += 1
        local_opt = True
        for (u, v) in zip(graph["vs_pos"][:-1], graph["vs_pos"][1:]):
                delta = delta_swap(u, v)
                u["pos"], v["pos"] = v["pos"], u["pos"]
                if delta < 0:
                        swap(graph, u, v)
                        graph["ELL"] += delta
                        local_opt = False
                        break
    graph["HLL"] = bars.total_HLL(graph)
    return graph

def adj_swap_pairs(graph, u, v):
     #Find a sequence of adjacent swap pairs equivalent to the swap of u, v
     inter_vs =  [graph["vs_pos"][q] for q in range(u["pos"]+1, v["pos"])]
     pairs = [(u, w) for w in inter_vs] + [(u, v)]
     pairs += [(w, v) for w in reversed(inter_vs)]
     return pairs

def compl_2opt(graph, maxiters = 10):
    #Iterate 2-OPT, swapping positions of any pair of vertices
    #Compute difference in ELL by performing a sequence of adjacent swaps that are equivalent to the arbitrary swap
    graph["ELL"] = total_ELL(graph)
    local_opt = False
    i = 0
    while not local_opt and i < maxiters:
        i += 1
        local_opt = True
        for u in graph["vs_pos"]:
            for v in graph["vs_pos"]:
                if u["pos"] >= v["pos"]:
                        continue
                swaps = adj_swap_pairs(graph, u, v)
                delta = np.sum([delta_swap(u1, v1) for (u1, v1) in swaps])
                u["pos"], v["pos"] = v["pos"], u["pos"]
                if delta < 0:
                        for (u1, v1) in swaps:
                            swap(graph,u1,v1)
                        graph["ELL"] += delta
                        local_opt = False
                        break
                
    graph["HLL"] = bars.total_HLL(graph)
    return graph