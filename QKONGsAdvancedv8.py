# -*- coding: utf-8 -*-
"""
=========================================================
KURAMOTO CHEVRON LAB Version 9.0
=========================================================

FULL INTERACTIVE VERSION

Features
--------
✓ Adjustable phase lag slider
✓ Stable percolation events
✓ Protected backbone edges
✓ Healing system
✓ Full restore button
✓ True desynchronization pulse
✓ Chevron generator
✓ Thermal synchronization maps
✓ Fractal turbulence fields
✓ Spectral turbulence visualization

NOW PRINTS ALL PLOTS IN A FOLDER FOR PUBLICATION!
=========================================================
"""

# =========================================================
# IMPORTS
# =========================================================

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk

import numpy as np
import networkx as nx

from math import sin, pi
from random import random

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

import collections
import os
# =========================================================
# CUSTOM THERMAL COLORMAP
# =========================================================

thermal_sync = LinearSegmentedColormap.from_list(
    "thermal_sync",
    [
        "#12001f",
        "#0033cc",
        "#00bb88",
        "#ffee00",
        "#ff55dd"
    ]
)

# =========================================================
# CONFIGURATION
# =========================================================

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
# GLOBALS
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

    return nx.convert_node_labels_to_integers(
        G,
        first_label=start_idx
    )

# =========================================================

def average_phase(nodes, theta_dict):

    phases = [theta_dict[n] for n in nodes]

    return np.arctan2(
        np.mean(np.sin(phases)),
        np.mean(np.cos(phases))
    )

# =========================================================

def kuramoto_order(nodes):

    phases = np.array([
        g.nodes[n]['theta']
        for n in nodes
    ])

    return np.abs(
        np.mean(np.exp(1j * phases))
    )

# =========================================================

def local_entropy(nodes):

    phases = np.array([
        g.nodes[n]['theta']
        for n in nodes
    ])

    bins = np.histogram(
        phases,
        bins=32,
        range=(0, 2*pi),
        density=True
    )[0]

    bins += 1e-12

    entropy = -np.sum(
        bins * np.log(bins)
    )

    entropy /= np.log(len(bins))

    return entropy

# =========================================================
# INITIALIZE NETWORK
# =========================================================

def initialize():

    global g
    global nextg

    global alice_nodes
    global charlie_nodes
    global bob_nodes

    g = nx.DiGraph()

    node_id = 0

    # =====================================================
    # ALICE GRID
    # =====================================================

    A = generate_grid_nodes(node_id, alice_size)

    node_id = max(A.nodes()) + 1

    alice_nodes = list(A.nodes())

    g.add_nodes_from(
        (
            n,
            {
                'theta': 2*pi*random(),

                'omega': np.random.normal(
                    1.0,
                    0.05
                )
            }
        )
        for n in alice_nodes
    )

    for u, v in A.edges():

        g.add_edge(
            u,
            v,
            weight=local_coupling
        )

        g.add_edge(
            v,
            u,
            weight=local_coupling
        )

    # =====================================================
    # CHARLIE GRID
    # =====================================================

    C = generate_grid_nodes(node_id, charlie_size)

    node_id = max(C.nodes()) + 1

    charlie_nodes = list(C.nodes())

    g.add_nodes_from(
        (
            n,
            {
                'theta': 2*pi*random(),

                'omega': np.random.normal(
                    1.0,
                    0.08
                )
            }
        )
        for n in charlie_nodes
    )

    for u, v in C.edges():

        g.add_edge(
            u,
            v,
            weight=local_coupling
        )

        g.add_edge(
            v,
            u,
            weight=local_coupling
        )

    # =====================================================
    # BOB GRID
    # =====================================================

    B = generate_grid_nodes(node_id, bob_size)

    bob_nodes = list(B.nodes())

    g.add_nodes_from(
        (
            n,
            {
                'theta': 2*pi*random(),

                'omega': np.random.normal(
                    1.0,
                    0.05
                )
            }
        )
        for n in bob_nodes
    )

    for u, v in B.edges():

        g.add_edge(
            u,
            v,
            weight=local_coupling
        )

        g.add_edge(
            v,
            u,
            weight=local_coupling
        )

    # =====================================================
    # META NODES
    # =====================================================

    g.add_node(
        "Alice",
        theta=0,
        omega=0
    )

    g.add_node(
        "Charlie",
        theta=0,
        omega=0
    )

    # =====================================================
    # INTER-GROUP COUPLING
    # =====================================================

    for c in charlie_nodes:

        g.add_edge(
            "Alice",
            c,
            weight=K_AC_forward
        )

        g.add_edge(
            c,
            "Alice",
            weight=K_AC_backward
        )

    for b in bob_nodes:

        g.add_edge(
            "Charlie",
            b,
            weight=K_CB_forward
        )

        g.add_edge(
            b,
            "Charlie",
            weight=K_CB_backward
        )

    nextg = g.copy()

# =========================================================
# DYNAMICS
# =========================================================

def update():

    global g
    global nextg
    global alpha_phase_lag

    current_theta = {

        n: g.nodes[n]['theta']

        for n in g.nodes()

        if isinstance(n, int)
    }

    # =====================================================
    # EDGE FLUCTUATIONS
    # =====================================================

    for u, v in list(g.edges()):

        if g.has_edge(u, v):

            if 'weight' in g[u][v]:

                g[u][v]['weight'] += np.random.normal(
                    0,
                    0.001
                )

                g[u][v]['weight'] = np.clip(
                    g[u][v]['weight'],
                    0.05,
                    3.0
                )

    # =====================================================
    # META PHASES
    # =====================================================

    g.nodes["Alice"]["theta"] = average_phase(
        alice_nodes,
        current_theta
    )

    g.nodes["Charlie"]["theta"] = average_phase(
        charlie_nodes,
        current_theta
    )

    # =====================================================
    # OSCILLATOR DYNAMICS
    # =====================================================

    for n in alice_nodes + charlie_nodes + bob_nodes:

        theta = g.nodes[n]['theta']

        omega = g.nodes[n]['omega']

        coupling = 0

        for j in g.predecessors(n):

            if not g.has_edge(j, n):
                continue

            w = g[j][n].get(
                'weight',
                1.0
            )

            phase_difference = (
                g.nodes[j]['theta'] - theta
            )

            nonlinear_term = (
                1
                + 0.15*np.cos(
                    3*phase_difference
                )
            )

            coupling += (

                w

                * sin(
                    phase_difference
                    - alpha_phase_lag
                )

                * nonlinear_term
            )

        # =================================================
        # ADAPTIVE TURBULENCE
        # =================================================

        adaptive_noise = (

            0.006

            * (
                1
                - abs(coupling)
                / (1 + abs(coupling))
            )
        )

        turbulence_noise = np.random.normal(
            0,
            adaptive_noise
        )

        nextg.nodes[n]['theta'] = (

            theta

            + (omega + coupling)*Dt

            + turbulence_noise
        )

    # =====================================================
    # ORDER PARAMETERS
    # =====================================================

    Ra = kuramoto_order(alice_nodes)
    Rc = kuramoto_order(charlie_nodes)
    Rb = kuramoto_order(bob_nodes)

    heatmap_history.append([
        Ra,
        Rc,
        Rb
    ])

    # =====================================================
    # ENTROPY
    # =====================================================

    Ea = local_entropy(alice_nodes)
    Ec = local_entropy(charlie_nodes)
    Eb = local_entropy(bob_nodes)

    entropy_drive = 0.03 * (
        1 - np.mean([Ra, Rc, Rb])
    )

    entropy_history.append([

        Ea + entropy_drive*np.random.random(),

        Ec + entropy_drive*np.random.random(),

        Eb + entropy_drive*np.random.random()
    ])

    # =====================================================
    # FRACTAL TURBULENCE
    # =====================================================

    if len(heatmap_history) > 8:

        H = np.array(heatmap_history)

        gradients = np.gradient(H[:,0])

        fractal_measure = np.mean(
            np.abs(gradients)
        )

        fractal_history.append(
            fractal_measure
        )

    g, nextg = nextg, g

# =========================================================
# SPECTRUM
# =========================================================

def compute_spectrum():

    H = g.to_undirected()

    nodes = list(H.nodes())

    idx = {
        n:i
        for i,n in enumerate(nodes)
    }

    N = len(nodes)

    A = np.zeros((N, N))

    for u, v in H.edges():

        i, j = idx[u], idx[v]

        w = H[u][v].get(
            'weight',
            1.0
        )

        tu = g.nodes[u]['theta'] \
            if isinstance(u, int) else 0

        tv = g.nodes[v]['theta'] \
            if isinstance(v, int) else 0

        phase = np.clip(
            np.cos(tv - tu),
            -0.95,
            0.95
        )

        A[i, j] = w * phase
        A[j, i] = w * phase

    D = np.diag(
        np.sum(np.abs(A), axis=1)
    )

    L = D - A

    eigs = np.sort(
        np.linalg.eigvalsh(L)
    )

    return eigs

# =========================================================
# CHEVRON GENERATOR
# =========================================================

def generate_chevron():

    global chevron_data

    print("Generating synchronization chevron...")

    detuning_values = np.linspace(
        -1.5,
        1.5,
        80
    )

    time_steps = 220

    chevron = np.zeros((
        len(detuning_values),
        time_steps
    ))

    for d_idx, delta in enumerate(detuning_values):

        initialize()

        for n in alice_nodes:

            g.nodes[n]['omega'] += delta

        for n in bob_nodes:

            g.nodes[n]['omega'] -= delta

        for t in range(time_steps):

            update()

            theta_A = average_phase(
                alice_nodes,
                {
                    n: g.nodes[n]['theta']
                    for n in alice_nodes
                }
            )

            theta_B = average_phase(
                bob_nodes,
                {
                    n: g.nodes[n]['theta']
                    for n in bob_nodes
                }
            )

            observable = np.cos(
                theta_A - theta_B
            )

            chevron[d_idx, t] = observable

    chevron_data = chevron

    print("Chevron generation complete.")

# =========================================================
# GUI
# =========================================================

class KuramotoGUI:

    def __init__(self, root):

        self.root = root

        root.title(
            "Kuramoto Chevron Lab"
        )

        self.running = False

        self.eigs_history = []

        # =================================================
        # CONTROL PANEL
        # =================================================

        control = tk.Frame(root)

        control.pack(
            side=tk.LEFT,
            fill=tk.Y
        )

        # =================================================
        # BUTTONS
        # =================================================

        self.start_btn = ttk.Button(
            control,
            text="Start",
            command=self.toggle
        )

        self.start_btn.pack(pady=5)

        self.chevron_btn = ttk.Button(
            control,
            text="Generate Chevron",
            command=self.show_chevron
        )

        self.chevron_btn.pack(pady=5)

        self.percolate_btn = ttk.Button(
            control,
            text="⚡ Percolation Event",
            command=self.percolate
        )

        self.percolate_btn.pack(pady=5)

        self.heal_btn = ttk.Button(
            control,
            text="Heal Network",
            command=self.heal_network
        )

        self.heal_btn.pack(pady=5)

        self.restore_btn = ttk.Button(
            control,
            text="Restore Network",
            command=self.restore_network
        )

        self.restore_btn.pack(pady=5)

        self.desync_btn = ttk.Button(
            control,
            text="Chaos Pulse",
            command=self.desync_pulse
        )

        self.desync_btn.pack(pady=5)

        self.export_btn = ttk.Button(
            control,
            text="Export Spectrum",
            command=self.export
        )

        self.export_btn.pack(pady=5)
        
# =================================================
# ARTICLE EXPORT
# =================================================
        self.article_btn = ttk.Button(
            control,
            text="Export Article Figures",
            command=self.export_article_figures
        )
        self.article_btn.pack(pady=5)
        
        # =================================================
        # ALPHA SLIDER
        # =================================================

        ttk.Label(
            control,
            text="Phase Lag α"
        ).pack(pady=(15,0))

        self.alpha_slider = tk.Scale(

            control,

            from_=0.0,

            to=1.57,

            resolution=0.01,

            orient=tk.HORIZONTAL,

            length=180,

            command=self.update_alpha
        )

        self.alpha_slider.set(
            alpha_phase_lag
        )

        self.alpha_slider.pack()

        # =================================================
        # FIGURE
        # =================================================

        self.fig = Figure(figsize=(10,18))

        self.axs = self.fig.subplots(5,1)

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=root
        )

        self.canvas.get_tk_widget().pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True
        )

        initialize()

        self.draw()

    # =====================================================
    # ALPHA UPDATE
    # =====================================================

    def update_alpha(self, value):

        global alpha_phase_lag

        alpha_phase_lag = float(value)

    # =====================================================
    # START / STOP
    # =====================================================

    def toggle(self):

        self.running = not self.running

        self.start_btn.config(
            text="Stop"
            if self.running
            else "Start"
        )

        if self.running:
            self.loop()

    # =====================================================
    # LOOP
    # =====================================================

    def loop(self):

        if self.running:

            update()

            eigs = compute_spectrum()

            self.eigs_history.append(eigs)

            if len(self.eigs_history) > 2000:
                self.eigs_history.pop(0)

            self.draw()

            self.root.after(
                40,
                self.loop
            )

    # =====================================================
    # STRUCTURED POSITIONS
    # =====================================================

    def structured_positions(self):

        pos = {}

        for i, n in enumerate(alice_nodes):

            x = i % alice_size[0]
            y = i // alice_size[0]

            pos[n] = (-7 + x, y)

        for i, n in enumerate(charlie_nodes):

            x = i % charlie_size[0]
            y = i // charlie_size[0]

            pos[n] = (x, y)

        for i, n in enumerate(bob_nodes):

            x = i % bob_size[0]
            y = i // bob_size[0]

            pos[n] = (7 + x, y)

        pos["Alice"] = (-3.5, 2)
        pos["Charlie"] = (3.5, 2)

        return pos

    # =====================================================
    # DRAW
    # =====================================================

    def draw(self):

        # =================================================
        # NETWORK
        # =================================================

        self.axs[0].clear()

        pos = self.structured_positions()

        node_colors = []

        for n in g.nodes():

            if n in alice_nodes:
                R = kuramoto_order(alice_nodes)

            elif n in charlie_nodes:
                R = kuramoto_order(charlie_nodes)

            elif n in bob_nodes:
                R = kuramoto_order(bob_nodes)

            else:
                R = 1

            node_colors.append(R)

        nx.draw(

            g,

            pos,

            ax=self.axs[0],

            node_size=420,

            node_color=node_colors,

            cmap=thermal_sync,

            edge_color="gray",

            width=1.2,

            arrows=False,

            with_labels=False
        )

        self.axs[0].set_title(
            f"Thermal Synchronization Network | α = {alpha_phase_lag:.2f}"
        )

        # =================================================
        # ORDER MAP
        # =================================================

        self.axs[1].clear()

        if len(heatmap_history) > 2:

            H = np.array(
                heatmap_history
            ).T

            self.axs[1].imshow(

                H,

                aspect='auto',

                origin='lower',

                cmap=thermal_sync,

                interpolation='gaussian',

                vmin=0,

                vmax=1
            )

            self.axs[1].set_yticks([0,1,2])

            self.axs[1].set_yticklabels([
                "Alice",
                "Charlie",
                "Bob"
            ])

            self.axs[1].set_title(
                "Order ↔ Chaos Thermal Evolution"
            )

        # =================================================
        # ENTROPY MAP
        # =================================================

        self.axs[2].clear()

        if len(entropy_history) > 2:

            E = np.array(
                entropy_history
            ).T

            self.axs[2].imshow(

                E,

                aspect='auto',

                origin='lower',

                cmap='magma',

                interpolation='gaussian'
            )

            self.axs[2].set_yticks([0,1,2])

            self.axs[2].set_yticklabels([
                "Alice",
                "Charlie",
                "Bob"
            ])

            self.axs[2].set_title(
                "Entropy / Turbulence Field"
            )

        # =================================================
        # FRACTAL FIELD
        # =================================================

        self.axs[3].clear()

        if len(fractal_history) > 5:

            F = np.array(fractal_history)

            self.axs[3].plot(
                F,
                color='#ff55dd',
                linewidth=1.5
            )

            self.axs[3].fill_between(
                range(len(F)),
                F,
                color='#ff55dd',
                alpha=0.3
            )

            self.axs[3].set_title(
                "Fractal Turbulence Measure"
            )

        # =================================================
        # SPECTRAL FIELD
        # =================================================

        self.axs[4].clear()

        if len(self.eigs_history) > 2:

            M = np.log1p(
                np.abs(
                    np.array(
                        self.eigs_history
                    ).T
                )
            )

            g1 = np.abs(
                np.gradient(M, axis=1)
            )

            g2 = np.abs(
                np.gradient(g1, axis=1)
            )

            spectral_gradient = g1 + 0.7*g2

            self.axs[4].imshow(

                spectral_gradient,

                aspect='auto',

                origin='lower',

                cmap='inferno',

                interpolation='bicubic'
            )

            self.axs[4].set_title(
                "Log-Spectrum Turbulence Field"
            )

        self.fig.tight_layout()

        self.canvas.draw()

    # =====================================================
    # CHEVRON
    # =====================================================

    def show_chevron(self):

        generate_chevron()

        if chevron_data is None:
            return

        win = tk.Toplevel(self.root)

        win.title(
            "Synchronization Chevron Map"
        )

        fig = Figure(figsize=(9,6))

        ax = fig.subplots()

        im = ax.imshow(

            chevron_data,

            aspect='auto',

            origin='lower',

            cmap='magma',

            interpolation='bicubic'
        )

        ax.set_xlabel(
            "Interaction Time"
        )

        ax.set_ylabel(
            "Detuning Index"
        )

        ax.set_title(
            "Kuramoto Synchronization Chevron"
        )

        fig.colorbar(
            im,
            ax=ax,
            label='cos(θA - θB)'
        )

        canvas = FigureCanvasTkAgg(
            fig,
            master=win
        )

        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )

        canvas.draw()

    # =====================================================
    # PERCOLATION
    # =====================================================

    def percolate(self):

        global g

        edges = list(g.edges())

        removable = []

        protected_nodes = {"Alice", "Charlie"}

        for u, v in edges:

            if u in protected_nodes or v in protected_nodes:
                continue

            removable.append((u, v))

        if len(removable) == 0:
            return

        remove_count = int(
            percolation_strength * len(removable)
        )

        remove_count = max(1, remove_count)

        chosen_indices = np.random.choice(
            len(removable),
            remove_count,
            replace=False
        )

        for idx in chosen_indices:

            u, v = removable[idx]

            if g.has_edge(u, v):

                g.remove_edge(u, v)

        print(
            f"Percolation removed {remove_count} edges."
        )

    # =====================================================
    # HEAL NETWORK
    # =====================================================

    def heal_network(self, reconnect_prob=0.03):

        global g

        nodes = (
            alice_nodes
            + charlie_nodes
            + bob_nodes
        )

        added = 0

        for u in nodes:

            for v in nodes:

                if u == v:
                    continue

                if random() < reconnect_prob:

                    if not g.has_edge(u, v):

                        g.add_edge(

                            u,

                            v,

                            weight=np.random.uniform(
                                0.5,
                                2.0
                            )
                        )

                        added += 1

        print(
            f"Healed network with {added} new edges."
        )

    # =====================================================
    # FULL RESTORE
    # =====================================================

    def restore_network(self):

        global heatmap_history
        global entropy_history
        global fractal_history

        print("Restoring original network...")

        initialize()

        heatmap_history.clear()
        entropy_history.clear()
        fractal_history.clear()

        self.eigs_history.clear()

        self.draw()

        print(
            "Network fully restored."
        )

    # =====================================================
    # TRUE DESYNCHRONIZATION PULSE
    # =====================================================

    def desync_pulse(self):

        for n in (
            alice_nodes
            + charlie_nodes
            + bob_nodes
        ):

            g.nodes[n]['theta'] += np.random.normal(
                0,
                2.5
            )

        print(
            "Chaos pulse injected."
        )






# =====================================================
# EXPORT
# =====================================================

    def export(self):

        eigs = compute_spectrum()

        plt.figure(figsize=(8, 5))

        plt.plot(eigs, 'o-')

        plt.title("Spectrum Snapshot")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.show()

    # =====================================================
    # ARTICLE EXPORT
    # =====================================================

    def export_article_figures(self):

        folder = "article_figures"
        os.makedirs(folder, exist_ok=True)

        print("\nGenerating publication figures...\n")

        # -------------------------------------------------
        # NETWORK FIGURE
        # -------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = self.structured_positions()

        node_colors = []

        for n in g.nodes():

            if n in alice_nodes:
                R = kuramoto_order(alice_nodes)

            elif n in charlie_nodes:
                R = kuramoto_order(charlie_nodes)

            elif n in bob_nodes:
                R = kuramoto_order(bob_nodes)

            else:
                R = 1

            node_colors.append(R)

        nx.draw(
            g,
            pos,
            ax=ax,
            node_size=420,
            node_color=node_colors,
            cmap=thermal_sync,
            edge_color="gray",
            arrows=False,
            with_labels=False
        )

        ax.set_title(
            f"Thermal Synchronization Network (α={alpha_phase_lag:.2f})"
        )

        plt.tight_layout()
        plt.savefig(
            os.path.join(folder, "network_sync.png"),
            dpi=600,
            bbox_inches="tight"
        )
        plt.close()

        # -------------------------------------------------
        # ORDER HEATMAP
        # -------------------------------------------------
        if len(heatmap_history) > 2:

            fig, ax = plt.subplots(figsize=(10, 4))

            H = np.array(heatmap_history).T

            im = ax.imshow(
                H,
                aspect='auto',
                origin='lower',
                cmap=thermal_sync,
                interpolation='gaussian',
                vmin=0,
                vmax=1
            )

            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Alice", "Charlie", "Bob"])
            ax.set_title("Order–Chaos Thermal Evolution")

            fig.colorbar(im)

            plt.tight_layout()
            plt.savefig(
                os.path.join(folder, "order_heatmap.png"),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close()

        # -------------------------------------------------
        # ENTROPY MAP
        # -------------------------------------------------
        if len(entropy_history) > 2:

            fig, ax = plt.subplots(figsize=(10, 4))

            E = np.array(entropy_history).T

            im = ax.imshow(
                E,
                aspect='auto',
                origin='lower',
                cmap='magma',
                interpolation='gaussian'
            )

            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["Alice", "Charlie", "Bob"])
            ax.set_title("Entropy / Turbulence Field")

            fig.colorbar(im)

            plt.tight_layout()
            plt.savefig(
                os.path.join(folder, "entropy_map.png"),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close()

        # -------------------------------------------------
        # FRACTAL TURBULENCE
        # -------------------------------------------------
        if len(fractal_history) > 5:

            fig, ax = plt.subplots(figsize=(8, 4))

            F = np.array(fractal_history)

            ax.plot(F, color="#ff55dd", linewidth=2)
            ax.fill_between(range(len(F)), F, alpha=0.3, color="#ff55dd")

            ax.set_title("Fractal Turbulence Measure")

            plt.tight_layout()
            plt.savefig(
                os.path.join(folder, "fractal_measure.png"),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close()

        # -------------------------------------------------
        # SPECTRAL FIELD
        # -------------------------------------------------
        if len(self.eigs_history) > 2:

            fig, ax = plt.subplots(figsize=(10, 4))

            M = np.log1p(np.abs(np.array(self.eigs_history).T))

            g1 = np.abs(np.gradient(M, axis=1))
            g2 = np.abs(np.gradient(g1, axis=1))

            spectral_gradient = g1 + 0.7 * g2

            im = ax.imshow(
                spectral_gradient,
                aspect='auto',
                origin='lower',
                cmap='inferno',
                interpolation='bicubic'
            )

            ax.set_title("Log-Spectrum Turbulence Field")

            fig.colorbar(im)

            plt.tight_layout()
            plt.savefig(
                os.path.join(folder, "spectral_field.png"),
                dpi=600,
                bbox_inches="tight"
            )
            plt.close()

        # -------------------------------------------------
        # CHEVRON
        # -------------------------------------------------
        global chevron_data

        if chevron_data is None:
            print("Generating Chevron...")
            generate_chevron()

        fig, ax = plt.subplots(figsize=(10, 6))

        im = ax.imshow(
            chevron_data,
            aspect='auto',
            origin='lower',
            cmap='magma',
            interpolation='bicubic'
        )

        ax.set_xlabel("Interaction Time")
        ax.set_ylabel("Detuning Index")
        ax.set_title("Kuramoto Synchronization Chevron")

        fig.colorbar(im, ax=ax, label=r'$\cos(\theta_A-\theta_B)$')

        plt.tight_layout()
        plt.savefig(
            os.path.join(folder, "chevron_map.png"),
            dpi=600,
            bbox_inches="tight"
        )
        plt.close()

        # -------------------------------------------------
        # SPECTRUM SNAPSHOT
        # -------------------------------------------------
        eigs = compute_spectrum()

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(eigs, 'o-', linewidth=2)
        ax.set_title("Network Laplacian Spectrum")
        ax.set_xlabel("Eigenvalue Index")
        ax.set_ylabel("Eigenvalue")
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(folder, "spectrum_snapshot.png"),
            dpi=600,
            bbox_inches="tight"
        )
        plt.close()

        print("\n" + "=" * 60)
        print("ARTICLE FIGURES EXPORTED")
        print("=" * 60)
        print(folder)
        print("=" * 60)
# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = KuramotoGUI(root)

    root.mainloop()