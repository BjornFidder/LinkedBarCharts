import numpy as np
from bars import total_HLL, swap, total_crossings
from blocks import baseline
import time

def adj_2opt(graph, iters=50):
    #Iterate 2-OPT, only swapping the positions of adjacent bars, using before and after a baseline (random) block order
    graph["HLL"] = total_HLL(graph)
    graph = baseline(graph)
    
    i = 0
    while i < iters:
        i += 1
        for (u, v) in zip(graph["vs_pos"][:-1], graph["vs_pos"][1:]):
                swap(graph, u, v)
                delta_HLL= total_HLL(graph) - graph["HLL"]
                VLL = graph["VLL"]
                graph = baseline(graph)
                delta_VLL = graph["VLL"] - VLL
                if delta_HLL + delta_VLL < 0:
                        graph["HLL"] += delta_HLL
                        break
                else:
                      swap(graph, u, v)
                      baseline(graph)
    return graph

def compl_2opt(graph, iters=50):
    #Iterate 2-OPT, swapping the position of any pair of bars, using before and after a baseline (random) block order
    graph["HLL"] = total_HLL(graph)
    graph = baseline(graph)
    
    i = 0
    while i < iters:
        i += 1
        for u in graph["vs_pos"]:
            for v in graph["vs_pos"]:
                if u == v:
                        continue
                swap(graph, u, v)
                delta_HLL= total_HLL(graph) - graph["HLL"]
                VLL = graph["VLL"]
                graph = baseline(graph)
                delta_VLL = graph["VLL"] - VLL
                if delta_HLL + delta_VLL < 0:
                        graph["HLL"] += delta_HLL
                        break
                else:
                      swap(graph, u, v)
                      baseline(graph)
    return graph

def repeat_alg(alg_blocks, alg_bars, alg_barsblocks, graphs, n_swaps=5):
    HLLs = []
    crossings = []
    times = []
    VLLs = []

    for graph in graphs:
        
        t1 = time.time()
        graph = alg_blocks(alg_barsblocks(alg_bars(graph), n_swaps))
        times.append(time.time() - t1)
        
        ne = max(graph.ecount(), 1) #average out by number of edges
        HLLs.append(graph["HLL"] / ne)
        crossings.append(total_crossings(graph) / ne)
        VLLs.append(graph["VLL"] / ne)

    return HLLs, crossings, times, VLLs