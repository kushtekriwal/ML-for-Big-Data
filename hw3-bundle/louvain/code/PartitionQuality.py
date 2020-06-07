import networkx as nx
import numpy as np
import pandas as pd
# pip install python-louvain
import community
import matplotlib.pyplot as plt


def read_network(data):
    g = nx.read_edgelist(data, create_using=nx.Graph())
    return g

def read_true_community(data):
    with open(data) as f:
        c = [line.rstrip().split('\t') for line in f]
    return c


def community_modularity(coms, g):
    if type(g) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")
    inc = 0
    deg = 0
    links = g.size()
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in g:
        if node not in coms:
            continue
        com = coms[node]
        deg += g.degree(node)
        for neighbor, dt in g[node].items():
            if neighbor not in coms:
                continue
            if neighbor == node:
                # YOUR CODE HERE
                inc += 0
            else:
                # YOUR CODE HERE
                inc += 1
    m = g.size()

    # YOUR CODE HERE
    mod = (inc / (2 * links)) - ((deg * deg) / (2 * links * 2 * links))
    return mod


def modularity(g, coms):
    part = {}
    ids = 0
    for n in coms:
        part[n] = ids
    mod = community_modularity(part, g)
    return mod

def density(coms):
    # YOUR CODE HERE
    den = (2 * len(coms.edges())) / (len(coms.nodes()) * (len(coms.nodes()) - 1))
    return den


def cut_ratio(g, coms):
    ns = len(coms.nodes())
    edges_outside = 0
    for nod in coms.nodes():
    	# YOUR CODE HERE
        for neighbor, dt in g[nod].items(): 
            if neighbor not in coms:
                edges_outside += 1
                    
    try:
        # YOUR CODE HERE
        ratio = (edges_outside) / (ns * (len(g.nodes()) - ns))
        
    except:
        return 0
    return ratio


def pquality_summary(graph, partition):
    mods, conds, dens, crs = [], [], [], []
    for cs in partition:
        if len(cs) > 1:
            #print(cs)
            community = graph.subgraph(cs)
            mods.append(modularity(graph, community))
            dens.append(density(community))
            crs.append(cut_ratio(graph,community))
            # Uncomment the next 4 lines for the sanity check:
            if cs == partition[0]:
                print(mods[-1])
                print(crs[-1])
                print(dens[-1])
    return [mods, crs, dens]



G = read_network('../data/youtube_net.txt')
C = read_true_community('../data/youtube_community_top1000.txt')
len_C = np.array([len(x) for x in C])
partition = C
mods, crs, dens = pquality_summary(G, partition)
mods_sort, den_sort_mods = zip(*sorted(zip(mods, dens), reverse=True))
crs_sort, den_sort_crs = zip(*sorted(zip(crs, dens), reverse=False))
den_sort_mods_avg = [np.mean(den_sort_mods[0:(i+1)]) for i in range(len(den_sort_mods))]
den_sort_crs_avg = [np.mean(den_sort_crs[0:(i+1)]) for i in range(len(den_sort_crs))]
ax = plt.gca()
ax.set_xlim(10**0, 10**3)
ax.set_xscale('log')
ax.plot(np.array([i for i in range(1000)]), np.array(den_sort_mods_avg), color='green', label='Modularity')
ax.plot(np.array([i for i in range(1000)]), np.array(den_sort_crs_avg), color='blue', label='Cut Ratio')
plt.title('Density')
plt.legend()
plt.xlabel('rank')
plt.ylabel('score')
plt.savefig("density.jpg")