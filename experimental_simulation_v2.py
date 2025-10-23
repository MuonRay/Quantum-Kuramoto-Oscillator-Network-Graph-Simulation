# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:12:42 2025

@author: ektop
"""

# -*- coding: utf-8 -*-
"""
Kuramoto Chain Network Simulation (v2)
--------------------------------------
Four-cluster version:
Alice, Bob, Dee (grids) connected to Charlie (small-world core).

Created on Oct 23, 2025
@author: ektop
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import sin, pi
from random import uniform, random
import collections
from scipy.sparse.linalg import eigsh

# ========================
# CONFIGURATION
# ========================

alice_size = (3, 3)
charlie_size = (2, 2)
bob_size = (3, 3)
dee_size = (3, 3)

# Coupling strengths
K_intra = 1.0
K_CA_forward = 0.8
K_CA_backward = 0.4
K_CB_forward = 0.8
K_CB_backward = 0.4
K_CD_forward = 0.8
K_CD_backward = 0.4

Dt = 0.05
tau_steps = 5
rewire_prob = 0.3

asymmetric_weights = True
delay_directional = True

# ========================
# GLOBAL VARIABLES
# ========================

g = None
nextg = None
alice_nodes = []
charlie_nodes = []
bob_nodes = []
dee_nodes = []

theta_history = collections.deque(maxlen=tau_steps + 1)
phase_log = []
order_param_log = []

# ========================
# HELPER FUNCTIONS
# ========================

def generate_grid_nodes(start_idx, shape):
    G = nx.grid_2d_graph(*shape)
    G = nx.convert_node_labels_to_integers(G, first_label=start_idx)
    return G

def average_phase(nodes, theta_dict):
    phases = [theta_dict[n] for n in nodes]
    return np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))

def kuramoto_order_parameter(theta_dict):
    phases = np.array(list(theta_dict.values()))
    r = np.abs(np.mean(np.exp(1j * phases)))
    return r

# ========================
# INITIALIZATION
# ========================

def initialize():
    """Initialize the 4-cluster Kuramoto network."""
    global g, nextg, alice_nodes, charlie_nodes, bob_nodes, dee_nodes
    global theta_history, phase_log, order_param_log

    g = nx.DiGraph()
    node_id = 0

    # --- Alice Grid ---
    A = generate_grid_nodes(node_id, alice_size)
    node_id = max(A.nodes()) + 1
    alice_nodes = list(A.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05, 0.05)}) for n in alice_nodes)
    g.add_edges_from(A.edges())
    g.add_edges_from([(v,u) for u,v in A.edges()])

    # --- Charlie Small-world ---
    C_size = charlie_size[0]*charlie_size[1]
    k_val = min(4, C_size - 1)
    if k_val % 2 != 0:
        k_val -= 1
    if k_val < 2:
        C = nx.path_graph(C_size)
    else:
        C = nx.watts_strogatz_graph(n=C_size, k=k_val, p=rewire_prob)
    C = nx.convert_node_labels_to_integers(C, first_label=node_id)
    node_id = max(C.nodes()) + 1
    charlie_nodes = list(C.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05, 0.05)}) for n in charlie_nodes)
    g.add_edges_from(C.edges())
    g.add_edges_from([(v,u) for u,v in C.edges()])

    # --- Bob Grid ---
    B = generate_grid_nodes(node_id, bob_size)
    node_id = max(B.nodes()) + 1
    bob_nodes = list(B.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05, 0.05)}) for n in bob_nodes)
    g.add_edges_from(B.edges())
    g.add_edges_from([(v,u) for u,v in B.edges()])

    # --- Dee Grid ---
    D = generate_grid_nodes(node_id, dee_size)
    node_id = max(D.nodes()) + 1
    dee_nodes = list(D.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05, 0.05)}) for n in dee_nodes)
    g.add_edges_from(D.edges())
    g.add_edges_from([(v,u) for u,v in D.edges()])

    # --- Add Aggregated Nodes ---
    g.add_node('Alice', theta=0.0, omega=0.0)
    g.add_node('Charlie', theta=0.0, omega=0.0)
    g.add_node('Bob', theta=0.0, omega=0.0)
    g.add_node('Dee', theta=0.0, omega=0.0)

    # --- Connect Charlie to clusters ---
    for a in alice_nodes:
        g.add_edge('Charlie', a, weight=K_CA_forward)
        g.add_edge(a, 'Charlie', weight=(K_CA_backward if asymmetric_weights else K_CA_forward))
    for b in bob_nodes:
        g.add_edge('Charlie', b, weight=K_CB_forward)
        g.add_edge(b, 'Charlie', weight=(K_CB_backward if asymmetric_weights else K_CB_forward))
    for d in dee_nodes:
        g.add_edge('Charlie', d, weight=K_CD_forward)
        g.add_edge(d, 'Charlie', weight=(K_CD_backward if asymmetric_weights else K_CD_forward))

    nextg = g.copy()
    theta_history.clear()
    phase_log.clear()
    order_param_log.clear()

# ========================
# SIMULATION STEP
# ========================

def update():
    """Perform one simulation step."""
    global g, nextg, theta_history, phase_log, order_param_log
    global K_CA_forward, K_CA_backward, K_CB_forward, K_CB_backward, K_CD_forward, K_CD_backward

    for a in alice_nodes:
        g.edges['Charlie', a]['weight'] = K_CA_forward
        g.edges[a, 'Charlie']['weight'] = (K_CA_backward if asymmetric_weights else K_CA_forward)
    for b in bob_nodes:
        g.edges['Charlie', b]['weight'] = K_CB_forward
        g.edges[b, 'Charlie']['weight'] = (K_CB_backward if asymmetric_weights else K_CB_forward)
    for d in dee_nodes:
        g.edges['Charlie', d]['weight'] = K_CD_forward
        g.edges[d, 'Charlie']['weight'] = (K_CD_backward if asymmetric_weights else K_CD_forward)

    current_theta = {n: g.nodes[n]['theta'] for n in g.nodes() if isinstance(n, int)}
    theta_history.append(current_theta.copy())
    if len(theta_history) < tau_steps + 1:
        theta_delayed = current_theta
    else:
        theta_delayed = theta_history[0]

    # Update aggregated phases
    g.nodes['Alice']['theta'] = average_phase(alice_nodes, current_theta)
    g.nodes['Bob']['theta']   = average_phase(bob_nodes, current_theta)
    g.nodes['Dee']['theta']   = average_phase(dee_nodes, current_theta)
    g.nodes['Charlie']['theta'] = average_phase(charlie_nodes, current_theta)

    for n in alice_nodes + charlie_nodes + bob_nodes + dee_nodes:
        theta_i = g.nodes[n]['theta']
        omega_i = g.nodes[n]['omega']
        neighbors = list(g.neighbors(n))
        coupling_sum = 0
        for j in neighbors:
            if j == 'Charlie':
                if delay_directional and n in (bob_nodes + dee_nodes):
                    theta_j = average_phase(charlie_nodes, theta_delayed)
                else:
                    theta_j = g.nodes[j]['theta']
                weight = g.edges[j, n]['weight']
            else:
                theta_j = g.nodes[j]['theta']
                weight = 1.0
            coupling_sum += weight * sin(theta_j - theta_i)

        deg = max(1, len(neighbors))
        nextg.nodes[n]['theta'] = theta_i + (omega_i + coupling_sum / deg) * Dt

    avg_A = average_phase(alice_nodes, current_theta)
    avg_C = average_phase(charlie_nodes, current_theta)
    avg_B = average_phase(bob_nodes, current_theta)
    avg_D = average_phase(dee_nodes, current_theta)
    phase_log.append((avg_A, avg_C, avg_B, avg_D))
    order_param_log.append(kuramoto_order_parameter(current_theta))
    g, nextg = nextg, g

# ========================
# VISUALIZATION
# ========================

def layout_group(nodes, x_offset, shape):
    rows, cols = shape
    pos = {}
    for idx, node in enumerate(nodes):
        r = idx // cols
        c = idx % cols
        pos[node] = (x_offset + c, -r)
    return pos

def draw_network(ax):
    ax.clear()
    pos = {}
    pos.update(layout_group(alice_nodes, -15, alice_size))
    pos.update(layout_group(charlie_nodes, 0, (1, len(charlie_nodes))))
    pos.update(layout_group(bob_nodes, 10, bob_size))
    pos.update(layout_group(dee_nodes, 15, dee_size))

    pos['Alice'] = (-18, 0)
    pos['Charlie'] = (0, 0)
    pos['Bob'] = (8, 0)
    pos['Dee'] = (18, 0)

    # Draw clusters
    nx.draw_networkx_nodes(g.subgraph(alice_nodes), pos, node_size=300, node_color='tab:red', ax=ax)
    nx.draw_networkx_nodes(g.subgraph(charlie_nodes), pos, node_size=300, node_color='tab:green', ax=ax)
    nx.draw_networkx_nodes(g.subgraph(bob_nodes), pos, node_size=300, node_color='tab:blue', ax=ax)
    nx.draw_networkx_nodes(g.subgraph(dee_nodes), pos, node_size=300, node_color='gold', ax=ax)

    nx.draw_networkx_edges(g.subgraph(alice_nodes), pos, ax=ax, edge_color='red')
    nx.draw_networkx_edges(g.subgraph(charlie_nodes), pos, ax=ax, edge_color='green')
    nx.draw_networkx_edges(g.subgraph(bob_nodes), pos, ax=ax, edge_color='blue')
    nx.draw_networkx_edges(g.subgraph(dee_nodes), pos, ax=ax, edge_color='gold')

    # Inter-group edges
    for a in alice_nodes:
        nx.draw_networkx_edges(g, pos, edgelist=[('Charlie', a)], edge_color='gray', arrows=True, ax=ax)
        nx.draw_networkx_edges(g, pos, edgelist=[(a, 'Charlie')], edge_color='gray', arrows=True, ax=ax)
    for b in bob_nodes:
        nx.draw_networkx_edges(g, pos, edgelist=[('Charlie', b)], edge_color='gray', arrows=True, ax=ax)
        nx.draw_networkx_edges(g, pos, edgelist=[(b, 'Charlie')], edge_color='gray', arrows=True, ax=ax)
    for d in dee_nodes:
        nx.draw_networkx_edges(g, pos, edgelist=[('Charlie', d)], edge_color='gray', arrows=True, ax=ax)
        nx.draw_networkx_edges(g, pos, edgelist=[(d, 'Charlie')], edge_color='gray', arrows=True, ax=ax)

    # Labels
    for lbl, color in [('Alice', 'red'), ('Charlie', 'green'), ('Bob', 'blue'), ('Dee', 'gold')]:
        ax.text(pos[lbl][0], pos[lbl][1] + 1.0, lbl, color=color, fontsize=12, fontweight='bold', ha='center')

    ax.axis('off')
    ax.set_title("Kuramoto 4-Cluster Network")

def draw_phase_circles(ax, phase_vals):
    ax.clear()
    labels = ['Alice', 'Charlie', 'Bob', 'Dee']
    colors = ['tab:red', 'tab:green', 'tab:blue', 'gold']

    for idx, label in enumerate(labels):
        ax.plot(np.cos(phase_vals[:, idx]), np.sin(phase_vals[:, idx]), label=label, color=colors[idx])
        ax.scatter(np.cos(phase_vals[-1, idx]), np.sin(phase_vals[-1, idx]), color=colors[idx], s=100)

    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.legend()
    ax.grid(True)
    ax.set_title("Phase Circle Evolution")

def plot_diagnostics():
    phase_arr = np.array(phase_log)
    order_arr = np.array(order_param_log)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    time = np.arange(len(phase_arr)) * Dt

    axs[0].plot(time, phase_arr[:, 0], label='Alice', color='tab:red')
    axs[0].plot(time, phase_arr[:, 1], label='Charlie', color='tab:green')
    axs[0].plot(time, phase_arr[:, 2], label='Bob', color='tab:blue')
    axs[0].plot(time, phase_arr[:, 3], label='Dee', color='gold')
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Average Phase (rad)")
    axs[0].legend()
    axs[0].set_title("Average Phases Over Time")

    axs[1].plot(time, order_arr, color='purple', label='Order Parameter')
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Order Parameter (r)")
    axs[1].legend()
    axs[1].set_title("Synchronization Over Time")

    plt.tight_layout()
    plt.show()
