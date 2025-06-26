import matplotlib.pyplot as plt
import numpy as np
from blocks import inter_vertices

def edge_height(v, edge):
    return edge["h_s"] if edge.source_vertex == v else edge["h_t"]

def draw_links(graph, x_pos):
    
    for edge in graph.es():
        xs = x_pos[edge.source_vertex["pos"]]
        xt = x_pos[edge.target_vertex["pos"]]
        ys = edge[f"h_{edge.source}"] + edge["weight"] / 2
        yt = edge[f"h_{edge.target}"] + edge["weight"] / 2

        dxs = 0.4 + 0.4*(1-ys / edge.source_vertex["height"])
        dxt = 0.4 + 0.4*(1-yt / edge.target_vertex["height"])
        yshift = 0

        xs2 = xs + dxs if xt > xs else xs - dxs
        xt2 = max(xt - dxt,xs2) if xt > xs else min(xt + dxt, xs2)

        inter_heights = [v["height"] for v in inter_vertices(edge, graph)]

        imax = np.argmax([ys,yt]+inter_heights)
        if imax == 0:
            xmax = xs2
            ymax = ys
        elif imax == 1:
            xmax = xt2
            ymax = yt
        else:
            pos_l = min(edge.source_vertex["pos"], edge.target_vertex["pos"]) 
            xmax = x_pos[pos_l + imax-1]
            ymax = inter_heights[imax-2]+yshift

        plt.plot([xs, xs2, xs2, xmax, xt2, xt2, xt], [ys, ys, ymax, ymax, ymax, yt, yt], '-', color='black')
                
def bar_chart(graph):

    bottoms = np.zeros(graph.vcount())
    heights = np.array([np.zeros(np.max(graph.degree())) for i in graph.vs()])

    for v in graph.vs():
        for (j, edge) in enumerate(v["links"]):
            heights[v["pos"]][j] = edge["weight"]

    x_pos = 1+ np.arange(0, 2*len(graph.vs()), step=2)
    for hs in heights.transpose():
        plt.bar(x_pos, hs, color='deeppink', edgecolor='white', linewidth=2, bottom=bottoms)
        bottoms += hs

    draw_links(graph, x_pos)

    bar_labels = [str(v.index+1) for v in graph["vs_pos"]]
    
    plt.xticks(x_pos, bar_labels)

    if graph.vcount() > 20:
        plt.xticks([])

    plt.ylim(0, 1.1*np.max([v["height"] for v in graph.vs()]))