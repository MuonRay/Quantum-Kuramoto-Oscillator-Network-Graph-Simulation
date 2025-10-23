# -*- coding: utf-8 -*-
"""
Kuramoto Chain Network GUI
--------------------------
Separate frontend for simulation.
Make sure experimental_simulation.py is in the same folder.

Created on Mon Oct 20 19:29:40 2025
@author: ektop
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from experimental_simulation import (
    initialize, update, draw_network, draw_phase_circles, plot_diagnostics,
    phase_log, K_AC_forward, K_AC_backward, K_CB_forward, K_CB_backward
)

#from experimental_simulation_v2 import (
#    initialize, update, draw_network, draw_phase_circles, plot_diagnostics,
#    phase_log,
#    K_CA_forward, K_CA_backward,
#    K_CB_forward, K_CB_backward,
#    K_CD_forward, K_CD_backward
#)



class KuramotoGUI:
    def __init__(self, root):
        self.root = root
        root.title("Kuramoto Chain Network Simulator")
        self.running = False

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        plot_frame = tk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.start_button = ttk.Button(control_frame, text="Start Simulation", command=self.toggle_simulation)
        self.start_button.pack(pady=5)

        # Sliders
        self.create_slider(control_frame, "K_AC Forward", K_AC_forward, self.update_k_ac_forward)
        self.create_slider(control_frame, "K_AC Backward", K_AC_backward, self.update_k_ac_backward)
        self.create_slider(control_frame, "K_CB Forward", K_CB_forward, self.update_k_cb_forward)
        self.create_slider(control_frame, "K_CB Backward", K_CB_backward, self.update_k_cb_backward)

        self.diagnostics_button = ttk.Button(control_frame, text="Show Diagnostics", command=plot_diagnostics)
        self.diagnostics_button.pack(pady=20)

        # Plots
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
    def update_k_ac_forward(self, val):
        import experimental_simulation as sim
        sim.K_AC_forward = float(val)

    def update_k_ac_backward(self, val):
        import experimental_simulation as sim
        sim.K_AC_backward = float(val)

    def update_k_cb_forward(self, val):
        import experimental_simulation as sim
        sim.K_CB_forward = float(val)

    def update_k_cb_backward(self, val):
        import experimental_simulation as sim
        sim.K_CB_backward = float(val)

if __name__ == "__main__":
    root = tk.Tk()
    app = KuramotoGUI(root)
    root.mainloop()
