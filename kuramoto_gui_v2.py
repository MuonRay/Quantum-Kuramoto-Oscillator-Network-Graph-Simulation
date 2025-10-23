# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:18:24 2025

@author: ektop
"""

# -*- coding: utf-8 -*-
"""
Kuramoto Chain Network GUI (v2)
Works with experimental_simulation_v2.py
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from experimental_simulation_v2 import (
    initialize, update, draw_network, draw_phase_circles, plot_diagnostics,
    phase_log,
    K_CA_forward, K_CA_backward,
    K_CB_forward, K_CB_backward,
    K_CD_forward, K_CD_backward
)

class KuramotoGUI:
    def __init__(self, root):
        self.root = root
        root.title("Kuramoto 4-Cluster Network Simulator")
        self.running = False

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Start/Stop Button ---
        self.start_button = ttk.Button(control_frame, text="Start Simulation", command=self.toggle_simulation)
        self.start_button.pack(pady=5)

        # --- Coupling sliders ---
        self.create_slider(control_frame, "K_CA Forward (Charlie→Alice)", K_CA_forward, self.update_k_ca_forward)
        self.create_slider(control_frame, "K_CA Backward (Alice→Charlie)", K_CA_backward, self.update_k_ca_backward)
        self.create_slider(control_frame, "K_CB Forward (Charlie→Bob)", K_CB_forward, self.update_k_cb_forward)
        self.create_slider(control_frame, "K_CB Backward (Bob→Charlie)", K_CB_backward, self.update_k_cb_backward)
        self.create_slider(control_frame, "K_CD Forward (Charlie→Dee)", K_CD_forward, self.update_k_cd_forward)
        self.create_slider(control_frame, "K_CD Backward (Dee→Charlie)", K_CD_backward, self.update_k_cd_backward)

        self.diagnostics_button = ttk.Button(control_frame, text="Show Diagnostics", command=plot_diagnostics)
        self.diagnostics_button.pack(pady=20)

        # --- Plots ---
        self.fig_network, self.ax_network = plt.subplots(figsize=(6, 5))
        self.canvas_network = FigureCanvasTkAgg(self.fig_network, master=plot_frame)
        self.canvas_network.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig_phase, self.ax_phase = plt.subplots(figsize=(6, 5))
        self.canvas_phase = FigureCanvasTkAgg(self.fig_phase, master=plot_frame)
        self.canvas_phase.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        initialize()
        self.draw_plots()

    def create_slider(self, frame, label, value, command):
        ttk.Label(frame, text=label).pack(pady=5)
        slider = ttk.Scale(frame, from_=0.0, to=3.0, value=value, command=command)
        slider.pack()
        return slider

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
            self.root.after(50, self.run_loop)

    def draw_plots(self):
        self.ax_network.clear()
        draw_network(self.ax_network)
        self.canvas_network.draw()

        self.ax_phase.clear()
        if len(phase_log) > 1:
            draw_phase_circles(self.ax_phase, np.array(phase_log))
        self.canvas_phase.draw()

    # --- Slider callbacks ---
    def update_k_ca_forward(self, val):
        import experimental_simulation_v2 as sim
        sim.K_CA_forward = float(val)

    def update_k_ca_backward(self, val):
        import experimental_simulation_v2 as sim
        sim.K_CA_backward = float(val)

    def update_k_cb_forward(self, val):
        import experimental_simulation_v2 as sim
        sim.K_CB_forward = float(val)

    def update_k_cb_backward(self, val):
        import experimental_simulation_v2 as sim
        sim.K_CB_backward = float(val)

    def update_k_cd_forward(self, val):
        import experimental_simulation_v2 as sim
        sim.K_CD_forward = float(val)

    def update_k_cd_backward(self, val):
        import experimental_simulation_v2 as sim
        sim.K_CD_backward = float(val)


if __name__ == "__main__":
    root = tk.Tk()
    app = KuramotoGUI(root)
    root.mainloop()
