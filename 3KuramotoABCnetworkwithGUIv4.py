# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 19:29:40 2025

@author: ektop
"""


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import numpy as np
import collections
from math import sin, pi
from random import random, uniform
from scipy.sparse.linalg import eigsh  # <-- Fix for eigsh error


# --- Globals ---
K_AC_forward = 0.8
K_AC_backward = 0.4
K_CB_forward = 0.8
K_CB_backward = 0.4

# Graph & simulation variables
g = None
nextg = None
alice_nodes = []
charlie_nodes = []
bob_nodes = []

tau_steps = 5
Dt = 0.05
asymmetric_weights = True
delay_directional = True
rewire_prob = 0.3

theta_history = collections.deque(maxlen=tau_steps + 1)
phase_log = []
order_param_log = []

# -----------------
# Initialization
# -----------------
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

def initialize():
    global g, nextg, alice_nodes, charlie_nodes, bob_nodes
    global theta_history, phase_log, order_param_log

    g = nx.DiGraph()
    node_id = 0

    # Alice 3x3 grid
    A = generate_grid_nodes(node_id, (3,3))
    node_id = max(A.nodes()) + 1
    alice_nodes = list(A.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05,0.05)}) for n in alice_nodes)
    g.add_edges_from(A.edges())
    g.add_edges_from([(v,u) for u,v in A.edges()])

    # Charlie 2x2 small-world
    C_size = 4
    C = nx.watts_strogatz_graph(C_size, k=2, p=rewire_prob)
    C = nx.convert_node_labels_to_integers(C, first_label=node_id)
    node_id = max(C.nodes()) + 1
    charlie_nodes = list(C.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05,0.05)}) for n in charlie_nodes)
    g.add_edges_from(C.edges())
    g.add_edges_from([(v,u) for u,v in C.edges()])

    # Bob 3x3 grid
    B = generate_grid_nodes(node_id, (3,3))
    bob_nodes = list(B.nodes())
    g.add_nodes_from((n, {'theta': 2*pi*random(), 'omega': 1 + uniform(-0.05,0.05)}) for n in bob_nodes)
    g.add_edges_from(B.edges())
    g.add_edges_from([(v,u) for u,v in B.edges()])

    # Aggregate nodes
    g.add_node('Alice', theta=0.0, omega=0.0)
    g.add_node('Charlie', theta=0.0, omega=0.0)

    # Inter-subsystem edges with weights
    for c in charlie_nodes:
        g.add_edge('Alice', c, weight=K_AC_forward)
        g.add_edge(c, 'Alice', weight=K_AC_backward if asymmetric_weights else K_AC_forward)
    for b in bob_nodes:
        g.add_edge('Charlie', b, weight=K_CB_forward)
        g.add_edge(b, 'Charlie', weight=K_CB_backward if asymmetric_weights else K_CB_forward)

    nextg = g.copy()
    theta_history.clear()
    phase_log.clear()
    order_param_log.clear()

# -----------------
# Update step
# -----------------
def update():
    global g, nextg
    global phase_log, order_param_log

    # Update weights based on globals
    for c in charlie_nodes:
        g.edges['Alice', c]['weight'] = K_AC_forward
        g.edges[c, 'Alice']['weight'] = K_AC_backward if asymmetric_weights else K_AC_forward
    for b in bob_nodes:
        g.edges['Charlie', b]['weight'] = K_CB_forward
        g.edges[b, 'Charlie']['weight'] = K_CB_backward if asymmetric_weights else K_CB_forward

    current_theta = {n: g.nodes[n]['theta'] for n in g.nodes() if isinstance(n, int)}
    theta_history.append(current_theta.copy())

    if len(theta_history) < tau_steps + 1:
        theta_delayed = current_theta
    else:
        theta_delayed = theta_history[0]

    g.nodes['Alice']['theta'] = average_phase(alice_nodes, current_theta)
    g.nodes['Charlie']['theta'] = average_phase(charlie_nodes, current_theta)

    for n in alice_nodes + charlie_nodes + bob_nodes:
        theta_i = g.nodes[n]['theta']
        omega_i = g.nodes[n]['omega']
        neighbors = list(g.neighbors(n))

        coupling_sum = 0
        for j in neighbors:
            if j == 'Alice':
                theta_j = g.nodes[j]['theta']
                weight = g.edges[j, n]['weight']
            elif j == 'Charlie':
                if delay_directional and n in bob_nodes:
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

    # Record phases for plots
    avg_A = average_phase(alice_nodes, current_theta)
    avg_C = average_phase(charlie_nodes, current_theta)
    avg_B = average_phase(bob_nodes, current_theta)
    phase_log.append((avg_A, avg_C, avg_B))
    order_param_log.append(kuramoto_order_parameter(current_theta))

    g, nextg = nextg, g

# -----------------
# Drawing functions
# -----------------
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
    pos.update(layout_group(alice_nodes, -10, (3,3)))
    pos.update(layout_group(charlie_nodes, 0, (1,len(charlie_nodes))))
    pos.update(layout_group(bob_nodes, 10, (3,3)))
    pos['Alice'] = (-13, 0)
    pos['Charlie'] = (3, 0)

    nx.draw_networkx_nodes(g.subgraph(alice_nodes), pos, node_color='tab:red', ax=ax, label='Alice')
    nx.draw_networkx_nodes(g.subgraph(charlie_nodes), pos, node_color='tab:green', ax=ax, label='Charlie')
    nx.draw_networkx_nodes(g.subgraph(bob_nodes), pos, node_color='tab:blue', ax=ax, label='Bob')

    nx.draw_networkx_edges(g.subgraph(alice_nodes), pos, ax=ax, edge_color='red')
    nx.draw_networkx_edges(g.subgraph(charlie_nodes), pos, ax=ax, edge_color='green')
    nx.draw_networkx_edges(g.subgraph(bob_nodes), pos, ax=ax, edge_color='blue')

    # Inter-subsystem edges
    inter_edges = [(u, v) for u,v in g.edges() if (u in ['Alice', 'Charlie'] or v in ['Alice', 'Charlie'])]
    nx.draw_networkx_edges(g, pos, edgelist=inter_edges, ax=ax, edge_color='black', arrowstyle='->', arrowsize=15)

    labels = {n: str(n) for n in alice_nodes + charlie_nodes + bob_nodes}
    labels['Alice'] = 'Alice'
    labels['Charlie'] = 'Charlie'
    nx.draw_networkx_labels(g, pos, labels, font_size=9, ax=ax)

    ax.set_title("Kuramoto Network")
    ax.axis('off')

def draw_phase_circles(ax, phases_array):
    # phases_array shape: (time, 3) for Alice, Charlie, Bob averages
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title("Phase Circles: Alice (red), Charlie (green), Bob (blue)")
    ax.axis('off')

    colors = ['red', 'green', 'blue']
    labels = ['Alice', 'Charlie', 'Bob']

    if phases_array.shape[0] < 1:
        return

    for i in range(3):
        phase = phases_array[-1, i]
        x = np.cos(phase)
        y = np.sin(phase)
        ax.plot([0, x], [0, y], color=colors[i], linewidth=4, label=labels[i])
        ax.scatter(x, y, color=colors[i], s=100)
    ax.legend(loc='upper right')

# -----------------
# Diagnostics plotting (outside GUI)
# -----------------
def plot_diagnostics():
    if len(phase_log) == 0:
        print("No simulation data yet.")
        return

    phases = np.array(phase_log)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(phases[:,0], label='Alice')
    plt.plot(phases[:,1], label='Charlie')
    plt.plot(phases[:,2], label='Bob')
    plt.xlabel('Time step')
    plt.ylabel('Avg Phase (radians)')
    plt.legend()
    plt.title('Average Phases')

    plt.subplot(1,2,2)
    plt.plot(order_param_log, label='Order parameter r')
    plt.xlabel('Time step')
    plt.ylabel('r')
    plt.legend()
    plt.title('Kuramoto Order Parameter')

    plt.tight_layout()
    plt.show()

# Function to plot fiedler heatmap on a given matplotlib Axes:
def plot_fiedler_heatmap_on_axes(graph, ax):
    undirected_g = graph.to_undirected()
    L = nx.laplacian_matrix(undirected_g).astype(float)
    vals, vecs = eigsh(L, k=2, which='SM')
    fiedler_vec = vecs[:, 1]
    norm_vec = (fiedler_vec - fiedler_vec.min()) / (fiedler_vec.max() - fiedler_vec.min())
    nodes_sorted = [node for _, node in sorted(zip(norm_vec, graph.nodes()))]
    A = nx.to_numpy_array(graph, nodelist=nodes_sorted)

    ax.clear()
    heatmap = ax.imshow(A, cmap='hot', interpolation='nearest')
    ax.set_title('Adjacency Matrix Heatmap Sorted by Fiedler Vector')
    ax.set_xlabel('Nodes sorted by Fiedler vector')
    ax.set_ylabel('Nodes sorted by Fiedler vector')
    # Clear existing colorbars to avoid multiple colorbars stacking
    fig = ax.figure
    for cax in fig.axes:
        if cax != ax and isinstance(cax, plt.Axes):
            cax.remove()
    ax.figure.colorbar(heatmap, ax=ax, orientation='vertical')




#GUI here:


import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class KuramotoGUI:
    def __init__(self, root):
        self.root = root
        root.title("Kuramoto Chain Network Simulator")

        # Simulation control
        self.running = False

        # Main frames
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Control Panel ---
        self.start_button = ttk.Button(control_frame, text="Start Simulation", command=self.toggle_simulation)
        self.start_button.pack(pady=5)

        ttk.Label(control_frame, text="Coupling Strength K_AC Forward").pack(pady=5)
        self.K_AC_forward_slider = ttk.Scale(control_frame, from_=0.0, to=3.0, value=K_AC_forward, command=self.update_k_ac_forward)
        self.K_AC_forward_slider.pack()

        ttk.Label(control_frame, text="Coupling Strength K_AC Backward").pack(pady=5)
        self.K_AC_backward_slider = ttk.Scale(control_frame, from_=0.0, to=3.0, value=K_AC_backward, command=self.update_k_ac_backward)
        self.K_AC_backward_slider.pack()

        ttk.Label(control_frame, text="Coupling Strength K_CB Forward").pack(pady=5)
        self.K_CB_forward_slider = ttk.Scale(control_frame, from_=0.0, to=3.0, value=K_CB_forward, command=self.update_k_cb_forward)
        self.K_CB_forward_slider.pack()

        ttk.Label(control_frame, text="Coupling Strength K_CB Backward").pack(pady=5)
        self.K_CB_backward_slider = ttk.Scale(control_frame, from_=0.0, to=3.0, value=K_CB_backward, command=self.update_k_cb_backward)
        self.K_CB_backward_slider.pack()

        self.diagnostics_button = ttk.Button(control_frame, text="Show Diagnostics", command=plot_diagnostics)
        self.diagnostics_button.pack(pady=20)

        # --- Plot Panels ---
        self.fig_network, self.ax_network = plt.subplots(figsize=(6, 5))
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, master=plot_frame)
        self.canvas_network.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_phase, self.ax_phase = plt.subplots(figsize=(6, 5))
        self.canvas_phase = FigureCanvasTkAgg(self.fig_phase, master=plot_frame)
        self.canvas_phase.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Initialize the simulation
        initialize()

        # Example graph for heatmap (replace with your actual graph)
        self.graph_for_heatmap = nx.path_graph(10)

        # Draw Fiedler heatmap at startup inside phase axes
        plot_fiedler_heatmap_on_axes(self.graph_for_heatmap, self.ax_phase)
        self.canvas_phase.draw()

        # Draw initial network plot
        self.draw_plots()

    def toggle_simulation(self):
        if not self.running:
            self.running = True
            self.start_button.config(text="Stop Simulation")
            self.run_loop()
        else:
            self.running = False
            self.start_button.config(text="Start Simulation")

    def run_loop(self):
        if self.running:
            update()
            self.draw_plots()
            # Schedule next update in 50 ms
            self.root.after(50, self.run_loop)

    def draw_plots(self):
        # Draw network
        self.ax_network.clear()
        draw_network(self.ax_network)
        self.canvas_network.draw()

        # Draw phase circles if running and data exists, else show heatmap
        self.ax_phase.clear()
        if self.running and len(phase_log) > 1:
            draw_phase_circles(self.ax_phase, np.array(phase_log))
        else:
            plot_fiedler_heatmap_on_axes(self.graph_for_heatmap, self.ax_phase)
        self.canvas_phase.draw()

    # Slider update callbacks
    def update_k_ac_forward(self, val):
        global K_AC_forward
        K_AC_forward = float(val)

    def update_k_ac_backward(self, val):
        global K_AC_backward
        K_AC_backward = float(val)

    def update_k_cb_forward(self, val):
        global K_CB_forward
        K_CB_forward = float(val)

    def update_k_cb_backward(self, val):
        global K_CB_backward
        K_CB_backward = float(val)

if __name__ == "__main__":
    root = tk.Tk()
    app = KuramotoGUI(root)
    root.mainloop()