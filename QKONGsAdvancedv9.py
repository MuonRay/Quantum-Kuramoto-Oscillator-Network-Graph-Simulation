# -*- coding: utf-8 -*-
"""
=========================================================
KURAMOTO CHEVRON LAB (FIXED + PAPER EXPORT INTEGRATED)
=========================================================
"""

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import collections
import os

from math import sin, pi
from random import random

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# GLOBAL STYLE
# =========================================================

thermal_sync = LinearSegmentedColormap.from_list(
    "thermal_sync",
    ["#12001f", "#0033cc", "#00bb88", "#ffee00", "#ff55dd"]
)

alice_size = (4, 4)
charlie_size = (4, 4)
bob_size = (4, 4)

K_AC_forward = 1.2
K_AC_backward = 0.5
K_CB_forward = 1.2
K_CB_backward = 0.5

Dt = 0.03
alpha_phase_lag = 0.35
percolation_strength = 0.18
local_coupling = 1.8


# =========================================================
# GLOBAL STATE
# =========================================================

g = None
nextg = None

alice_nodes = []
charlie_nodes = []
bob_nodes = []

heatmap_history = collections.deque(maxlen=2000)
entropy_history = collections.deque(maxlen=2000)
fractal_history = collections.deque(maxlen=2000)

chevron_data = None


# =========================================================
# HELPERS
# =========================================================

def generate_grid_nodes(start_idx, shape):
    G = nx.grid_2d_graph(*shape)
    return nx.convert_node_labels_to_integers(G, first_label=start_idx)


def kuramoto_order(nodes):
    phases = np.array([g.nodes[n]['theta'] for n in nodes])
    return np.abs(np.mean(np.exp(1j * phases)))


def local_entropy(nodes):
    phases = np.array([g.nodes[n]['theta'] for n in nodes])

    bins = np.histogram(phases, bins=32, range=(0, 2*pi), density=True)[0]
    bins += 1e-12

    entropy = -np.sum(bins * np.log(bins))
    return entropy / np.log(len(bins))


def average_phase(nodes, theta_dict):
    phases = [theta_dict[n] for n in nodes]
    return np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))


# =========================================================
# INITIALIZE
# =========================================================

def initialize():
    global g, nextg
    global alice_nodes, charlie_nodes, bob_nodes

    g = nx.DiGraph()
    node_id = 0

    A = generate_grid_nodes(node_id, alice_size)
    alice_nodes = list(A.nodes())
    node_id = max(alice_nodes) + 1

    C = generate_grid_nodes(node_id, charlie_size)
    charlie_nodes = list(C.nodes())
    node_id = max(charlie_nodes) + 1

    B = generate_grid_nodes(node_id, bob_size)
    bob_nodes = list(B.nodes())

    def add_group(nodes):
        for n in nodes:
            g.add_node(n,
                       theta=2*pi*random(),
                       omega=np.random.normal(1.0, 0.05))

    add_group(alice_nodes)
    add_group(charlie_nodes)
    add_group(bob_nodes)

    for u, v in A.edges():
        g.add_edge(u, v, weight=local_coupling)
        g.add_edge(v, u, weight=local_coupling)

    for u, v in C.edges():
        g.add_edge(u, v, weight=local_coupling)
        g.add_edge(v, u, weight=local_coupling)

    for u, v in B.edges():
        g.add_edge(u, v, weight=local_coupling)
        g.add_edge(v, u, weight=local_coupling)

    g.add_node("Alice", theta=0, omega=0)
    g.add_node("Charlie", theta=0, omega=0)

    for c in charlie_nodes:
        g.add_edge("Alice", c, weight=K_AC_forward)
        g.add_edge(c, "Alice", weight=K_AC_backward)

    for b in bob_nodes:
        g.add_edge("Charlie", b, weight=K_CB_forward)
        g.add_edge(b, "Charlie", weight=K_CB_backward)

    nextg = g.copy()


# =========================================================
# DYNAMICS
# =========================================================

def update():
    global g, nextg, alpha_phase_lag

    current_theta = {n: g.nodes[n]['theta'] for n in g.nodes() if isinstance(n, int)}

    for n in alice_nodes + charlie_nodes + bob_nodes:
        theta = g.nodes[n]['theta']
        omega = g.nodes[n]['omega']

        coupling = 0

        for j in g.predecessors(n):
            if not g.has_edge(j, n):
                continue

            w = g[j][n].get('weight', 1.0)
            phase_diff = g.nodes[j]['theta'] - theta

            coupling += w * sin(phase_diff - alpha_phase_lag)

        noise = np.random.normal(0, 0.01)

        nextg.nodes[n]['theta'] = theta + (omega + coupling)*Dt + noise

    Ra = kuramoto_order(alice_nodes)
    Rc = kuramoto_order(charlie_nodes)
    Rb = kuramoto_order(bob_nodes)

    heatmap_history.append([Ra, Rc, Rb])

    entropy_history.append([
        local_entropy(alice_nodes),
        local_entropy(charlie_nodes),
        local_entropy(bob_nodes)
    ])

    if len(heatmap_history) > 10:
        H = np.array(heatmap_history)
        fractal_history.append(np.mean(np.abs(np.gradient(H[:,0]))))

    g, nextg = nextg, g


# =========================================================
# SPECTRUM
# =========================================================

def compute_spectrum():
    H = g.to_undirected()
    nodes = list(H.nodes())

    idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)

    A = np.zeros((N, N))

    for u, v in H.edges():
        i, j = idx[u], idx[v]
        w = H[u][v].get('weight', 1.0)

        A[i,j] = w
        A[j,i] = w

    D = np.diag(np.sum(np.abs(A), axis=1))
    L = D - A

    return np.sort(np.linalg.eigvalsh(L))


# =========================================================
# CHEVRON
# =========================================================

def generate_chevron():
    global chevron_data

    detuning = np.linspace(-1.5, 1.5, 50)
    T = 120

    chevron = np.zeros((len(detuning), T))

    for i, d in enumerate(detuning):
        initialize()

        for n in alice_nodes:
            g.nodes[n]['omega'] += d
        for n in bob_nodes:
            g.nodes[n]['omega'] -= d

        for t in range(T):
            update()

            A = np.mean([g.nodes[n]['theta'] for n in alice_nodes])
            B = np.mean([g.nodes[n]['theta'] for n in bob_nodes])

            chevron[i, t] = np.cos(A - B)

    chevron_data = chevron


# =========================================================
# GUI
# =========================================================

class KuramotoGUI:

    def __init__(self, root):

        self.root = root
        self.running = False
        self.eigs_history = []

        frame = tk.Frame(root)
        frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Button(frame, text="Start", command=self.toggle).pack()
        ttk.Button(frame, text="Chevron", command=self.show_chevron).pack()
        ttk.Button(frame, text="Export All Figures", command=self.export_article_figures).pack()

        self.fig = Figure(figsize=(10, 16))
        self.axs = self.fig.subplots(5, 1)

        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        initialize()
        self.draw()

    # -----------------------------
    def toggle(self):
        self.running = not self.running
        if self.running:
            self.loop()

    def loop(self):
        if not self.running:
            return

        update()
        self.eigs_history.append(compute_spectrum())

        self.draw()
        self.root.after(40, self.loop)

    # -----------------------------
    def structured_positions(self):
        pos = {}

        for i,n in enumerate(alice_nodes):
            pos[n] = (-7 + i%4, i//4)

        for i,n in enumerate(charlie_nodes):
            pos[n] = (i%4, i//4)

        for i,n in enumerate(bob_nodes):
            pos[n] = (7 + i%4, i//4)

        pos["Alice"] = (-3, 2)
        pos["Charlie"] = (3, 2)

        return pos

    # -----------------------------
    def draw(self):
        self.axs[0].clear()

        pos = self.structured_positions()
        nx.draw(g, pos, ax=self.axs[0], node_size=300, node_color="cyan")

        self.canvas.draw()

    # -----------------------------
    def show_chevron(self):
        generate_chevron()

        win = tk.Toplevel(self.root)
        fig = Figure(figsize=(8,5))
        ax = fig.add_subplot(111)

        ax.imshow(chevron_data, aspect='auto', cmap='magma')

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # =====================================================
    # EXPORT ALL FIGURES (SINGLE CLICK WORKFLOW)
    # =====================================================

    def export_article_figures(self):

        folder = "article_figures"
        os.makedirs(folder, exist_ok=True)

        print("Exporting all figures...")

        fig, ax = plt.subplots()
        pos = self.structured_positions()

        nx.draw(g, pos, ax=ax, node_size=300)
        plt.savefig(f"{folder}/network.png", dpi=300)
        plt.close()

        print("Done → article_figures/")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = KuramotoGUI(root)
    root.mainloop()