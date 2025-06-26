#%%

import numpy as np
import time
from bars import total_crossings

def inter_vertices(edge, graph):
    #Find the set of vertices with positions between those of the edge source and target
    p1, p2 = np.sort([edge.source_vertex["pos"], edge.target_vertex["pos"]])
    return [graph["vs_pos"][q] for q in range(p1+1, p2)]

def VLL(edge, graph):

    h1 = edge[f"h_{edge.source}"] + edge["weight"] / 2
    h2 = edge[f"h_{edge.target}"] + edge["weight"] / 2

    inter_heights = [v["height"] for v in inter_vertices(edge, graph)]

    if not inter_heights:
        return np.abs(h1 - h2)
    h_mid = np.max(inter_heights) #max intermediate height
    
    if h_mid < h1 and h_mid < h2:
        return np.abs(h1 - h2)
    else:
        return np.abs(h1 - h_mid) + np.abs(h2 - h_mid)
    
def total_VLL(graph):
    return np.sum([VLL(edge, graph) for edge in graph.es()])

def minmaxedge(m, edge):
    return m(edge.source_vertex["pos"], edge.target_vertex["pos"])

def baseline(graph, uniform_prob = False):
    
    for v in graph.vs():
        edges_L = sorted([edge for edge in v.incident() if minmaxedge(min, edge) < v["pos"]],
                          key=lambda edge: -minmaxedge(min, edge))
        edges_R = sorted([edge for edge in v.incident() if minmaxedge(max, edge) > v["pos"]], 
                          key=lambda edge: minmaxedge(max, edge))

        v["height"] = 0
        v["links"] = []
        while edges_L or edges_R:
            
            if uniform_prob:
                edge = edges_L.pop(0) if (np.random.randint(0, 1) or not edges_R) and edges_L else edges_R.pop(0)
            else:
                edge = edges_L.pop(0) if np.random.randint(0,1) < len(edges_L) / (len(edges_L) + len(edges_R)) else edges_R.pop(0)

            edge[f"h_{v.index}"] = v["height"]
                
            v["links"].append(edge)
            v["height"] += edge["weight"]

    graph["VLL"] = total_VLL(graph) 
    return graph

def baseline2(graph):
    return baseline(graph, True)

def is_left_edge(edge, v):
    if edge.source_vertex == v:
        return (edge.source_vertex["pos"] > edge.target_vertex["pos"])
    else:
        return (edge.source_vertex["pos"] < edge.target_vertex["pos"])

def swap(e1, e2, v):

    dw = e2["weight"] - e1["weight"] # difference in weights
    e1[f"h_{v.index}"], e2[f"h_{v.index}"] = e2[f"h_{v.index}"] + dw, e1[f"h_{v.index}"]
    i1, i2 = v["links"].index(e1), v["links"].index(e2)
    v["links"][i1], v["links"][i2] = v["links"][i2], v["links"][i1]

def delta_swap(e1, e2, v, graph):
    if (is_left_edge(e1, v) and is_left_edge(e2, v)) or \
       (not is_left_edge(e1, v) and not is_left_edge(e2, v)):
        return 0
    
    VLL0 = VLL(e1, graph) + VLL(e2, graph)
    swap(e1, e2, v)#e1[f"h_{v.index}"], e2[f"h_{v.index}"] = e2[f"h_{v.index}"], e1[f"h_{v.index}"]
    VLL1 = VLL(e1, graph) + VLL(e2, graph)
    swap(e2, e1, v)#e1[f"h_{v.index}"], e2[f"h_{v.index}"] = e2[f"h_{v.index}"], e1[f"h_{v.index}"]
    return VLL1 - VLL0

def adj_2opt(graph):
    graph = baseline(graph)

    local_opt = False
    while not local_opt:
        local_opt = True
        for v in graph.vs():
            for (e1, e2) in zip(v["links"][:-1], v["links"][1:]):
                
                delta = delta_swap(e1, e2, v, graph)
                if delta < 0:
                    swap(e1, e2, v)
                    graph["VLL"] += delta
                    local_opt = False
                    break
            else:
                continue
            break

    return graph

def iter_DP(graph, iters=None):
    if iters is None:
        iters = 5*graph.vcount()
    
    graph = baseline(graph)

    VLLs = []
    
    for _ in range(iters):

        v = np.random.choice(graph.vs())

        edges_L = [edge for edge in v["links"] if is_left_edge(edge, v)]
        edges_R = [edge for edge in v["links"] if not is_left_edge(edge, v)]
        
        nL = len(edges_L)
        nR = len(edges_R)

        if nL == 0 or nR == 0:
            continue

        heights_L = np.cumsum([0]+[edge["weight"] for edge in edges_L])
        heights_R = np.cumsum([0]+[edge["weight"] for edge in edges_R])

        VLL_old = np.sum([VLL(edge, graph) for edge in v.incident()])

        VLL_table = np.zeros((nL+1, nR+1))
        links_table = np.empty((nL+1, nR+1), dtype=object)

        links_table[0, 0] = []
        for L in range(nL):
            edges_L[L][f"h_{v.index}"] = heights_L[L]
            VLL_table[L+1, 0] = VLL_table[L, 0] + VLL(edges_L[L], graph)
            links_table[L+1, 0] = edges_L[:L+1]
        
        for R in range(nR):
            edges_R[R][f"h_{v.index}"] = heights_R[R]
            VLL_table[0, R+1] = VLL_table[0, R] + VLL(edges_R[R], graph)
            links_table[0, R+1] = edges_R[:R+1]

        for L in range(nL):
            for R in range(nR):
                
                edges_L[L][f"h_{v.index}"] = heights_L[L] + heights_R[R+1]
                edges_R[R][f"h_{v.index}"] = heights_L[L+1] + heights_R[R]

                VLL_split = [VLL_table[L, R+1] + VLL(edges_L[L], graph),
                             VLL_table[L+1, R] + VLL(edges_R[R], graph)]

                VLL_table[L+1, R+1] = np.min(VLL_split)

                if np.argmin(VLL_split) == 0:
                    links_table[L+1, R+1] = links_table[L, R+1] + [edges_L[L]]
                else:
                    links_table[L+1, R+1] = links_table[L+1, R] + [edges_R[R]]

        graph["VLL"] += VLL_table[nL, nR] - VLL_old
        v["links"] = links_table[nL, nR]

        height = 0
        for edge in v["links"]:
            edge[f"h_{v.index}"] = height
            height += edge["weight"]

        VLLs.append(graph["VLL"])

    return graph

def repeat_alg(alg_blocks, alg_bars, graphs):

    HLLs = []
    crossings = []
    times = []
    VLLs = []

    for graph in graphs:
        
        t1 = time.time()
        graph = alg_blocks(alg_bars(graph))
        times.append(time.time() - t1)
        
        ne = max(graph.ecount(), 1) #average out by number of edges
        HLLs.append(graph["HLL"] / ne)
        crossings.append(total_crossings(graph) / ne)
        VLLs.append(graph["VLL"] / ne)

    return HLLs, crossings, times, VLLs




        
