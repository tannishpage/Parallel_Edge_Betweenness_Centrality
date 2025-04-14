import networkx as nx
import random as r


from collections import deque
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import py_random_state
import multiprocessing
import time
from pytictoc import TicToc

#Custom thread class
class SubProcess(multiprocessing.Process):
    
    def __init__(self, G, nodes, start_time, to_be_returned, betweenness, pid, verbose=False):
        """
        Initializing Process for computing single source shortest paths for each node

        Parameters:
            - G <NetworkX.Graph> : Graph to compute shortest path on
            - nodes <list> : List of nodes to compute shortest paths for
            - start_time <float> : The time the parent process started to create the sub-processes
            - to_be_returned <multiprocessing.Queue> : A queue to store and return the final betweenness values
            - betweenness <dict> : Betweenness scores for nodes and edges, will be updated but won't affect parent process
        """
        multiprocessing.Process.__init__(self)
        self._killed = False
        self._to_be_returned = to_be_returned
        self._nodes = nodes
        self._timer = TicToc()
        self._start_time = start_time
        self._betweenness = betweenness
        self._G = G
        self._verbose = verbose
        self._pid = pid
        
        
    def run(self):
        """
        Run through all nodes and compute shortest path, update betweenness values.

        At the end add the updated betweenness to the queue
        """
        if self._verbose:
            self._timer.tic()
        for n in self._nodes: # Iterate through all nodes
            # Compute shortest path
            S, P, sigma, _ = _single_source_shortest_path_basic(self._G, n)
            # Update betweenness
            self._betweenness = self._accumulate_edges(self._betweenness, S, P, sigma, n)
        # Add betweenness to queue
        self._to_be_returned.put(self._betweenness)
        self._killed = True
        if self._verbose:
            self._timer.toc(f"Process {self._pid} Finished executing in")
    
    def _accumulate_edges(self, betweenness, S, P, sigma, s):
        """
        From Networkx, used to update the betweenness value
        """
        delta = dict.fromkeys(S, 0)
        while S:
            w = S.pop()
            coeff = (1 + delta[w]) / sigma[w]
            for v in P[w]:
                c = sigma[v] * coeff
                if (v, w) not in betweenness:
                    betweenness[(w, v)] += c
                else:
                    betweenness[(v, w)] += c
                delta[v] += c
            if w != s:
                betweenness[w] += delta[w]
        return betweenness

# Copy of Edge betweenness centrality and helper functions
@py_random_state(4)
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None, verbose=False):
    internal_timer = TicToc()
    betweenness = dict.fromkeys(G, 0.0)  # b[v]=0 for v in G
    # b[e]=0 for e in G.edges()
    betweenness.update(dict.fromkeys(G.edges(), 0.0))
    if k is None:
        nodes = list(G.nodes())
    else:
        nodes = seed.sample(list(G.nodes()), k)
    internal_timer.tic()
    # Creating a queue to store all the betweenness values computed by sup-processes
    accumulators = multiprocessing.Queue()
    start = 0
    number_of_workers = 16 # Change if necessary
    step_size = len(nodes)//number_of_workers
    end = start + step_size
    pid = 0
    # Store sup-processes to use them later
    sub_processes = []
    # To be used to tell order of operations
    start_time = time.time()
        
    internal_timer.tic()
    
    # Dividing work among workers
    while end < len(nodes):
        if verbose:
            print(f"Creating Sub-Process {pid} with {end - start} nodes")
        sub_processes.append(SubProcess(G, nodes[start:end], start_time, accumulators, betweenness, pid, verbose))
        pid += 1
        sub_processes[-1].start()
        start = end
        end = start + step_size

    # If there are still nodes left in the list that haven't been assigned
    if (start < len(nodes)) and (end >= len(nodes)):
        if verbose:
            print(f"Creating Sub-Process {pid} with {len(nodes) - start} nodes")
        sub_processes.append(SubProcess(G, nodes[start:len(nodes)], start_time, accumulators, betweenness, pid, verbose))
        sub_processes[-1].start()

    # Accumulate the data for betweenness
    all_betweenness = [accumulators.get() for b in range(len(sub_processes))]
    betweenness = all_betweenness[0]
    for b in all_betweenness[1:]:
        for key in b.keys():
            betweenness[key] += b[key]

    # rescaling
    for n in G:  # remove nodes to only return edges
        del betweenness[n]
    betweenness = _rescale_e(
        betweenness, len(G), normalized=normalized, directed=G.is_directed()
    )
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight)

    return betweenness


# helpers for betweenness centrality


def _single_source_shortest_path_basic(G, s):
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    D[s] = 0
    Q = deque([s])
    while Q:  # use BFS to find shortest paths
        v = Q.popleft()
        S.append(v)
        Dv = D[v]
        sigmav = sigma[v]
        for w in G[v]:
            if w not in D:
                Q.append(w)
                D[w] = Dv + 1
            if D[w] == Dv + 1:  # this is a shortest path, count paths
                sigma[w] += sigmav
                P[w].append(v)  # predecessors
    return S, P, sigma, D


def _single_source_dijkstra_path_basic(G, s, weight):
    weight = _weight_function(G, weight)
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    push(Q, (0, next(c), s, s))
    while Q:
        (dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = dist
        for w, edgedata in G[v].items():
            vw_dist = dist + weight(v, w, edgedata)
            if w not in D and (w not in seen or vw_dist < seen[w]):
                seen[w] = vw_dist
                push(Q, (vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_dist == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    return S, P, sigma, D

def _rescale_e(betweenness, n, normalized, directed=False, k=None):
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / (n * (n - 1))
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness


def _add_edge_keys(G, betweenness, weight=None):
    r"""Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    _weight = _weight_function(G, weight)

    edge_bc = dict.fromkeys(G.edges, 0.0)
    for u, v in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[(u, v)] / len(keys)
        for k in keys:
            edge_bc[(u, v, k)] = bc

    return edge_bc

def best_edge(G):
    betweenness = edge_betweenness_centrality(G)
    return max(betweenness, key=betweenness.get)
    
def pop_centrality_best_edge(G):
    # Get the vertex with the smallest degree, check all the connections
    # return the edge to the vertex with the highest degree
    pass
    
def girvan_newman(G, most_valuable_edge=None):
    # If the graph is already empty, simply return its connected
    # components.
    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:

        def most_valuable_edge(G):
            """Returns the edge with the highest betweenness centrality
            in the graph `G`.

            """
            # We have guaranteed that the graph is non-empty, so this
            # dictionary will never be empty.
            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)

    # The copy of G here must include the edge weight data.
    g = G.copy().to_undirected()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(nx.selfloop_edges(g))
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)


def _without_most_central_edges(G, most_valuable_edge):
    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        print("Removing Edge:", edge)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components

# Creating a random graph
if __name__ == "__main__":
    r.seed(9090)
    G = nx.Graph()
    random_graph_nodes = [x for x in range(0, 500)]
    random_graph_nodes2 = [x for x in range(500, 1000)]

    for x in range(0, 100000):
        v1 = r.choice(random_graph_nodes)
        v2 = r.choice(random_graph_nodes)
        if v1 != v2:
            G.add_edge(v1, v2)
        
        v3 = r.choice(random_graph_nodes2)
        v4 = r.choice(random_graph_nodes2)
        if v3 != v4:
            G.add_edge(v3, v4)

    G.add_edge(v3, v2)
    print(v3, v2)
        

    print(f"Number of Nodes: {G.number_of_nodes()}\nNumber of Edges: {G.number_of_edges()}")

    # nx.draw(G)
    # plt.show()

    timer = TicToc()

    community_gen = girvan_newman(G, best_edge)
    print("Starting community detection")
    timer.tic()
    for com in community_gen:
        timer.toc()
        timer.tic()
        print("Number of Components:", len(com))
        break # Exiting after first iteration
