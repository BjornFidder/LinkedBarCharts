#%%
import gen, bars, blocks, vis, barsblocks, bars_heuristic
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from network_data import edge_density, teenage_friends, roll_call_votes, movies

path = 'C:/Users/bjorn/OneDrive - Universiteit Utrecht/UNI/Master/Jaar 2/Semester 2/Experimentation Project/'
colors = {'Baseline': "C0", 
          'Greedy incremental': "C3", 
          'Adjacent 2-OPT': "C1", 
          'Complete 2-OPT': "C2",
          'Heuristic Adjacent 2-OPT': "C4",
          'Heuristic Complete 2-OPT': "C6",
          '2-OPT': "C1",
          'Iterative DP': "C2",
          'HLL': "C0",
          'VLL': "C1",
          'Total': "C2",
          'Baseline, Baseline':"C0",
          'Baseline, 2-OPT':"C3", 
          'Adjacent 2-OPT, 2-OPT':"C1",
          'Complete 2-OPT, 2-OPT':"C2",
          }

def report(values):
    print(f"{np.round(np.average(values), 2)} +- {np.round(np.std(values), 2)}")

def plot_vals(ps, values, label=''):
    y = np.apply_along_axis(np.average, arr=values, axis=1)
    yerr = np.apply_along_axis(np.std, arr=values, axis=1)

    indices = np.argsort(ps)
    ps = ps[indices]
    y = y[indices]
    yerr = yerr[indices]

    plt.errorbar(ps, y, yerr = yerr, alpha=.75, fmt=':', capsize=3, capthick=1, label=label, color=colors[label])
    y_low = [y - e for y, e in zip(y, yerr)]
    y_high = [y + e for y, e in zip(y, yerr)]
    plt.fill_between(ps, y_low, y_high, alpha=.25, color=colors[label])

#%% Bars
algs = [\
        bars.baseline, 
        bars.gr_incr, 
        bars.adj_2opt, 
        bars.compl_2opt
        ]
labels = [\
          'Baseline', 
          'Greedy incremental', 
          'Adjacent 2-OPT', 
          'Complete 2-OPT'
          ]
file_names = [s+'50' for s in \
              [\
               'baseline', 
               'gr_incr', 
               'adj_2opt', 
               'compl_2opt'
               ]]

repeats = 20
n = 50
ps = np.linspace(0.02, 0.2, num=10)
# ns = np.array([10, 20, 30, 40, 50])
# p = 0.1
data = 'DataRandom/'
graph_gen = gen.random_graph
prop = 0
load = True

for alg, label, file_name in zip(algs, labels, file_names):

    graphs = [[graph_gen(n, p) for i in range(repeats)] for p in ps]

    if load:
        vals= np.load(path+data+file_name+".npy")
    else:
        vals = np.array([bars.repeat_alg(alg, gs) for gs in tqdm(graphs)])
        np.save(path+data+file_name+'.npy', vals)

    vals_prop = [v[prop] for v in vals]
    plot_vals(ps, vals_prop, label=label)

plt.xlabel('Connection probability $p$', fontsize=12)
#plt.xlabel('Neighborhood radius $r$')
#plt.xlabel('Number of vertices $N$')
plt.xlim(0, plt.xlim()[1])  
plt.locator_params(axis='x', nbins=5)
plt.ylim(0, plt.ylim()[1])
plt.ylabel(['Average horizontal link length', 'Average number of crossings', 'Run-time (s)'][prop],
           fontsize = 12)
plt.legend()
plt.savefig(path+'Plots/plot.pdf', dpi=300)
plt.show()


#%% Blocks fixed
algs = [\
        bars.baseline, 
        bars.gr_incr, 
        bars.adj_2opt, 
        bars.compl_2opt
        ]

alg_blocks = blocks.adj_2opt
blocks_label = '2-OPT on blocks'

labels = [\
          'Baseline', 
          'Greedy incremental', 
          'Adjacent 2-OPT', 
          'Complete 2-OPT'
          ]
file_names = [s+"_2opt_50" for s in \
              [\
               'baseline', 
               'gr_incr', 
               'adj_2opt', 
               'compl_2opt'
               ]]

n = 50
ps = np.linspace(0.01, 0.1, num=10)
repeats = 20
data = 'DataRGG/'
graph_gen = gen.random_graph
load = True

for alg, label, file_name in zip(algs, labels, file_names):

    graphs = [[graph_gen(n, p) for i in range(repeats)] for p in ps]

    if load:
        vals= np.load(path+data+file_name+".npy")
    else:
        vals = np.array([blocks.repeat_alg(alg_blocks, alg, gs) for gs in tqdm(graphs)])
        np.save(path+data+file_name+'.npy', vals)

    vals_prop = [v[0]+v[3] for v in vals]
    plot_vals(ps, vals_prop, label=label)

plt.title(blocks_label)
plt.xlabel('Neighborhood radius $r$', fontsize=12)
plt.xlim(0, plt.xlim()[1])
plt.ylabel('Average total link length', fontsize=12)
plt.legend()
plt.savefig(path+'Plots/plot.pdf', dpi=300)
plt.show()

#%% Bars fixed
algs = [\
        blocks.baseline,
        blocks.adj_2opt,
        blocks.iter_DP
        ]

alg_bars = bars.compl_2opt
bars_label = 'Complete 2-OPT on bars'

labels = [\
          'Baseline', 
          '2-OPT', 
          'Iterative DP'
          ]
file_names = ["compl_2opt_"+s+"_50" for s in \
              [\
               'baseline', 
               '2opt', 
               'iterDP'
               ]]

n = 50
ps = np.linspace(0.01, 0.1, num=10)
repeats = 20
data = 'DataRGG/'
graph_gen = gen.random_geometric_graph
load = True

for alg, label, file_name in zip(algs, labels, file_names):

    graphs = [[graph_gen(n, p) for i in range(repeats)] for p in ps]

    if load:
        vals= np.load(path+data+file_name+".npy")
    else:
        vals = np.array([blocks.repeat_alg(alg, alg_bars, gs) for gs in tqdm(graphs)])
        np.save(path+data+file_name+'.npy', vals)

    vals_prop = [v[3] for v in vals] 
    plot_vals(ps, vals_prop, label=label)

plt.title(bars_label)
plt.xlabel('Neighborhood radius $r$', fontsize=12)
plt.xlim(0, plt.xlim()[1])
plt.ylabel('Average vertical link length', fontsize=12)
plt.legend()
plt.savefig(path+'Plots/plot.pdf', dpi=300)
plt.show()

#%% HLL and VLL

alg_bars = bars.compl_2opt

alg_blocks = bars.adj_2opt

title = "Bars: Complete 2-OPT, Blocks: 2-OPT"

labels = [\
          'Baseline', 
          '2-OPT', 
          'Iterative DP'
          ]
file_names = ["compl_2opt_"+s+"_50" for s in \
              [\
               'baseline', 
               '2opt', 
               'iterDP'
               ]]

n = 50
ps = np.linspace(0.01, 0.1, num=10)
repeats = 20
data = 'DataRandom/'
graph_gen = gen.random_graph
load = True

graphs = [[graph_gen(n, p) for i in range(repeats)] for p in ps]

if load:
    vals= np.load(path+data+file_name+".npy")
else:
    vals = np.array([blocks.repeat_alg(alg, alg_bars, gs) for gs in tqdm(graphs)])
    np.save(path+data+file_name+'.npy', vals)

vals_prop = [v[0] for v in vals]
plot_vals(ps, vals_prop, label='HLL')

vals_prop = [v[3] for v in vals] 
plot_vals(ps, vals_prop, label='VLL')

vals_prop = [v[0]+v[3] for v in vals]
plot_vals(ps, vals_prop, label='Total')

plt.title(title)
plt.xlabel('Connection probability $p$', fontsize=12)
plt.xlim(0, plt.xlim()[1])
plt.ylabel('Average link length', fontsize=12)
plt.legend()
plt.savefig(path+'Plots/plot.pdf', dpi=300)
plt.show()

#%% Run-times
algs_blocks = [\
        blocks.baseline,
        blocks.adj_2opt,
        blocks.iter_DP
        ]
algs_bars = [\
        bars.baseline, 
        bars.gr_incr, 
        bars.adj_2opt, 
        bars.compl_2opt
        ]

file_names_blocks =  [\
                        'baseline', 
                        '2opt', 
                        'iterDP'
                     ] 
file_names_bars = [\
                    'baseline_',
                    'gr_incr_',
                    'adj_2opt_', 
                    'compl_2opt_'
                  ]

n = 50
ps = np.linspace(0.01, 0.1, num=10)
repeats = 20
data = 'DataRandom/'
graph_gen = gen.random_graph
load = True

for alg_bar, fn_bar in zip(algs_bars, file_names_bars):
    for alg_block, fn_block in zip(algs_blocks, file_names_blocks):

        graphs = [[graph_gen(n, p) for i in range(repeats)] for p in ps]

        try:
            vals= np.load(path+data+fn_bar+fn_block+'_50.npy')
        except:
            vals = np.array([blocks.repeat_alg(alg_bar, alg_block, gs) for gs in tqdm(graphs)])
            np.save(path+data+fn_bar+fn_block+'_50.npy', vals)

        #print(fn_bar+fn_block)
        report([v[2][-1] for v in vals])


#%% Test DP
graph = gen.random_graph(50, 0.2)

graph, VLLs = blocks.iter_DP(bars.compl_2opt(graph), iters=50*10)

plt.plot(VLLs)

#%% Example bar chart visual
graph = gen.random_graph(10, 0.2)

blocks.adj_2opt(bars.compl_2opt(graph))

vis.bar_chart(graph)
plt.gca().yaxis.set_visible(False)
plt.xticks([])
plt.tight_layout()
plt.savefig(path+"Plots/plot.pdf", dpi=300, bbox_inches='tight', pad_inches=0)


#%% Bar chart visual
#graph = roll_call_votes(edge_density=0.1)[0]
graph = gen.random_graph(20, 0.1)

blocks.baseline(bars.baseline(graph))
fig, ax = plt.subplots(3, 1)
fig.set_figheight(8)

plt.axes(ax[0])
vis.bar_chart(graph)
plt.title('Baseline')

print(graph["HLL"], graph["VLL"])

blocks.baseline(bars.compl_2opt(graph))

plt.axes(ax[1])
vis.bar_chart(graph)
plt.title('Bars: Complete 2-OPT')

print(graph["HLL"], graph["VLL"])
 
blocks.adj_2opt(graph)

plt.axes(ax[2])
vis.bar_chart(graph)
plt.title('Blocks: 2-OPT')

print(graph["HLL"], graph["VLL"])

plt.tight_layout()
plt.savefig(path+"Plots/plot.pdf", dpi=300)


# %% Test gen

plt.figure()

ts = np.linspace(0, 1, num=30)
repeats = 20
ERs = []
RGGs = []
for t in tqdm(ts):
    ERs.append([gen.random_graph(50, t).ecount() for _ in range(repeats)])
    RGGs.append([gen.random_geometric_graph(50, t, dim=1).ecount() for _ in range(repeats)])
    
plot_vals(ts, ERs, label="Random graph")
plot_vals(ts, RGGs, label="Random geometric graph")
plt.xlabel("Parameter $p$, $r$")
plt.ylabel("Number of edges")
plt.legend()
plt.savefig(path+'Plots/plot.pdf', dpi=300)
plt.show()

#%% Visual barsblocks

graph = gen.random_graph(50, 0.1)
blocks.baseline(bars.baseline(graph))
fig, ax = plt.subplots(5, 1)
fig.set_figheight(20)
    
plt.axes(ax[0])
vis.bar_chart(graph)
plt.title('Baseline')

print(graph["HLL"], graph["VLL"])

blocks.baseline(bars.compl_2opt(graph))

plt.axes(ax[1])
vis.bar_chart(graph)
plt.title('Bars: Complete 2-OPT')

print(graph["HLL"], graph["VLL"])

barsblocks.compl_2opt(graph, 20)

plt.axes(ax[3])
vis.bar_chart(graph)
plt.title('Bars/blocks: 2-OPT')

print(graph["HLL"], graph["VLL"])
plt.tight_layout()

blocks.adj_2opt(graph)

plt.axes(ax[4])
vis.bar_chart(graph)
plt.title('Blocks: 2-OPT')

print(graph["HLL"], graph["VLL"])
plt.tight_layout()


#%% Visual ELL
graph = roll_call_votes()[1]
blocks.baseline(bars.baseline(graph))
fig, ax = plt.subplots(5, 1)
fig.set_figheight(20)
    
plt.axes(ax[0])
vis.bar_chart(graph)
plt.title('Baseline')

print(graph["HLL"], graph["VLL"], bars_heuristic.total_ELL(graph))

blocks.baseline(bars_heuristic.compl_2opt(graph, 50))

plt.axes(ax[1])
vis.bar_chart(graph)
plt.title('Heuristic complete bars + blocks baseline')

print(graph["HLL"], graph["VLL"], graph["ELL"])

blocks.adj_2opt(graph)

plt.axes(ax[2])
vis.bar_chart(graph)
plt.title('Heuristic compl bars + blocks baseline')

print(graph["HLL"], graph["VLL"], graph["ELL"])

blocks.baseline(bars.compl_2opt(graph))

plt.axes(ax[3])
vis.bar_chart(graph)
plt.title('Bars: Complete 2-OPT')

print(graph["HLL"], graph["VLL"], bars_heuristic.total_ELL(graph))

blocks.adj_2opt(graph)

plt.axes(ax[4])
vis.bar_chart(graph)
plt.title('Blocks: 2-OPT')

print(graph["HLL"], graph["VLL"], bars_heuristic.total_ELL(graph))
plt.tight_layout()


#%% Teenage friends

algs = [\
        bars.baseline, 
        bars.adj_2opt, 
        # bars_heuristic.adj_2opt,
        bars.compl_2opt,
        # bars_heuristic.compl_2opt
        ]

alg_blocks = blocks.adj_2opt
blocks_label = '2-OPT on blocks'

labels = [\
          'Baseline', 
          'Adjacent 2-OPT', 
        #   'Heuristic Adjacent 2-OPT',
          'Complete 2-OPT',
        #   'Heuristic Complete 2-OPT'
          ]
file_names = [s+"_2opt" for s in \
              [\
               'baseline',  
               'adj_2opt', 
            #    'heuristic_adj_2opt',
               'compl_2opt',
            #    'heuristic_compl_2opt'

               ]]

repeats = 20
data = 'DataTF/'
load = True

for alg, label, file_name in zip(algs, labels, file_names):

    graphs = np.array([teenage_friends() for i in range(repeats)]).transpose()
    ps = np.array([edge_density(gs[0]) for gs in graphs])

    if load:
        vals= np.load(path+data+file_name+".npy")
    else:
        vals = np.array([blocks.repeat_alg(alg_blocks, alg, gs) for gs in tqdm(graphs)])
        np.save(path+data+file_name+'.npy', vals)

    vals_prop = [v[0]+v[3] for v in vals]
    plot_vals(ps, vals_prop, label=label)

plt.title('Teenage friends')
plt.xlabel('Edge density $p$', fontsize=12)
plt.ylabel('Average total link length', fontsize=12)
plt.legend()
plt.savefig(path+'Plots/TF_blocks_TLL.pdf', dpi=300)
plt.show()

#%% Movies

algs = [\
        bars.baseline, 
        bars.adj_2opt, 
        # bars_heuristic.adj_2opt,
        bars.compl_2opt,
        # bars_heuristic.compl_2opt
        ]

alg_blocks = blocks.adj_2opt
blocks_label = '2-OPT on blocks'

labels = [\
          'Baseline', 
          'Adjacent 2-OPT', 
        #   'Heuristic Adjacent 2-OPT',
          'Complete 2-OPT',
        #   'Heuristic Complete 2-OPT'
          ]
file_names = [s+"_2opt" for s in \
              [\
               'baseline',  
               'adj_2opt', 
            #    'heuristic_adj_2opt',
               'compl_2opt',
            #    'heuristic_compl_2opt'

               ]]

repeats = 20
data = 'DataMovies/'
load = True

for alg, label, file_name in zip(algs, labels, file_names):

    graphs = np.array([movies(load_range=(1, 100), n_range=(40, 50)) for i in range(repeats)]).transpose()
    ps = np.array([edge_density(gs[0]) for gs in graphs])

    if load:
        vals= np.load(path+data+file_name+".npy")
    else:
        vals = np.array([blocks.repeat_alg(alg_blocks, alg, gs) for gs in tqdm(graphs)])
        np.save(path+data+file_name+'.npy', vals)

    vals_prop = [v[0]+v[3] for v in vals]
    plot_vals(ps, vals_prop, label=label)

plt.title('Movie actors')
plt.xlabel('Edge density $p$', fontsize=12)
plt.ylabel('Average total link length', fontsize=12)
plt.legend()
plt.savefig(path+'Plots/Movies_blocks_TLL.pdf', dpi=300)
plt.show()

#%% Roll Call Votes

algs = [\
        (bars.baseline, bars.baseline),
        (bars.baseline, blocks.adj_2opt),
        (bars.adj_2opt, blocks.adj_2opt),
        (bars.compl_2opt, blocks.adj_2opt),
        ]

labels = [\
          'Baseline, Baseline', 
          'Baseline, 2-OPT', 
          'Adjacent 2-OPT, 2-OPT',
          'Complete 2-OPT, 2-OPT',
          ]
file_names = ['baseline_baseline',  
               'baseline_2opt', 
               'adj_2opt_2opt',
               'compl_2opt_2opt'
               ]

repeats = 20
data = 'DataRCV/'
load = True
ps = np.linspace(0.01, 0.1, num=10)

for alg, label, file_name in zip(algs, labels, file_names):

    # graphs = np.array([[roll_call_votes(mat_max=0, edge_density=p)[0] for i in range(repeats)] for p in ps])
    
    if load:
        vals= np.load(path+data+file_name+".npy")
    else:
        vals = np.array([blocks.repeat_alg(alg[0], alg[1], gs) for gs in tqdm(graphs)])
        np.save(path+data+file_name+'.npy', vals)

    vals_prop = [v[0]+v[3] for v in vals]
    plot_vals(ps, vals_prop, label=label)

plt.xlabel('Edge density $p$')
plt.ylabel('Average total link length')
plt.legend()
plt.savefig(path+'Plots/RCV.pdf', dpi=300)
plt.show()


