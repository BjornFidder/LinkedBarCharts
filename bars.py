#%%
import numpy as np
import time

def HLL(edge):
    #Compute the horizontal link length corresponding to one edge
    return np.abs(edge.source_vertex["pos"] - edge.target_vertex["pos"])

def vertex_HLL(v):
     #horizontal link length of all edges adjacent to vertex v
     return np.sum([HLL(edge) for edge in v.incident()])

def total_HLL(graph):
    #Compute the total horizontal link length of a graph, based on the "pos" attributes of the vertices
    return np.sum([HLL(edge) for edge in graph.es()])

def total_crossings(graph):
    #Compute the total number of crossings 
    crossings = 0
    for edge1 in graph.es():
         for edge2 in graph.es():
              p1, q1 = np.sort([edge1.source_vertex["pos"], edge1.target_vertex["pos"]])
              p2, q2 = np.sort([edge2.source_vertex["pos"], edge2.target_vertex["pos"]])
              if p1 < p2 and p2 < q1 and q1 < q2:
                   crossings += 1
    return crossings

def repeat_alg(alg, graphs):
    #Apply the algorithm to each graph in the list of graphs,
    #keeping track of their horizontal link lengths, number of crossings and run-times.
    HLLs = []
    crossings = []
    times = []

    for graph in graphs:
        
        t1 = time.time()
        graph = alg(graph)
        times.append(time.time() - t1)
        
        ne = max(graph.ecount(), 1) #average out by number of edges
        HLLs.append(graph["HLL"] / ne)
        crossings.append(total_crossings(graph) / ne)

    return HLLs, crossings, times

def baseline(graph):
    #The baseline does not permute the bar order, so takes a random bar order.
    graph["HLL"] = total_HLL(graph)
    return graph

def swap(graph, u, v):
     #Swap the positions of vertices u and v
     u["pos"], v["pos"] = v["pos"], u["pos"]
     graph["vs_pos"][u["pos"]], graph["vs_pos"][v["pos"]] = graph["vs_pos"][v["pos"]], graph["vs_pos"][u["pos"]]
     
def delta_swap(u,v):
     #Compute the change in total HLL when swapping vertices u and v
     u["pos"], v["pos"] = v["pos"], u["pos"]
     delta = vertex_HLL(u) + vertex_HLL(v)
     u["pos"], v["pos"] = v["pos"], u["pos"]
     delta -= vertex_HLL(u) + vertex_HLL(v)
     return delta

def adj_2opt(graph):
    #Iterate 2-OPT, only swapping the positions of adjacent vertices  
    graph["HLL"] = total_HLL(graph)
    local_opt = False
    while not local_opt:
        local_opt = True
        for (u, v) in zip(graph["vs_pos"][:-1], graph["vs_pos"][1:]):
                delta = delta_swap(u,v)
                if delta < 0:
                        swap(graph, u,v)
                        graph["HLL"] += delta
                        local_opt = False
                        break
    return graph

def compl_2opt(graph):
    #Iterate 2-OPT, swapping positions of any pair of vertices
    graph["HLL"] = total_HLL(graph)
    local_opt = False
    while not local_opt:
        local_opt = True
        for u in graph["vs_pos"]:
            for v in graph["vs_pos"]:
                if u == v:
                        continue
                delta = delta_swap(u,v)
                if delta < 0:
                        swap(graph, u,v)
                        graph["HLL"] += delta
                        local_opt = False
                        break
    return graph


def gr_incr(graph):
    #Greedy incremental
    #Incrementally construct the bar order, by greedily placing next bar on the left or right
    graph["HLL"] = 0

    left = -1
    right = 1

    for v in graph.vs()[1:]:
        v["pos"] = left
        left_HLL = np.sum([HLL(edge) for edge in v.incident() if edge.target == v.index])
        
        v["pos"] = right
        right_HLL = np.sum([HLL(edge) for edge in v.incident() if edge.target == v.index])
            
        if right_HLL <= left_HLL:
            v["pos"] = right
            right += 1
            graph["HLL"] += right_HLL
        
        else:
            v["pos"] = left
            left -= 1
            graph["HLL"] += left_HLL

    #Shift positions s.t. starts at 0, and set position array accordingly
    min_pos = np.min([v["pos"] for v in graph.vs()])
    for v in graph.vs():
        v["pos"] -= min_pos
        graph["vs_pos"][v["pos"]] = v

    return graph

                
def sort_on_height(graph):
     
    graph["HLL"] = 0

    vs_sorted = sorted(graph.vs(),key = lambda v: np.sum([edge['weight'] for edge in v.incident()]))

    for (i, v) in enumerate(vs_sorted):
        v["pos"] = i
        graph["vs_pos"][i] = v

    graph["HLL"] = total_HLL(graph)

    return graph
    
    