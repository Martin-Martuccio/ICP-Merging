''' ### No more used ###
# here we have to load PLY files , performing ICP, merging models, and highlighting differences.
from io_handler import load_ply_as_point_cloud
from merging import compare_and_color, load_plydata, compute_transformation, apply_transformation
from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
'''
from SplatPLYHandler import SplatPLYHandler
import tkinter as tk
from tkinter import filedialog, messagebox
import os

class PLYProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("PLY Model Processor")
        
        # Variabili per i percorsi dei file
        self.file1_path = ""
        self.file2_path = ""
        self.output_path = os.path.abspath("../data/output/merged_model.ply")

        # Widget GUI
        self.label = tk.Label(master, text="Select two PLY files to process")
        self.label.grid(row=0, column=0, columnspan=3, pady=10)

        # Pulsante per il primo file
        self.btn_file1 = tk.Button(master, text="Load the first model", command=self.load_file1)
        self.btn_file1.grid(row=1, column=0, padx=5)
        self.label_file1 = tk.Label(master, text="No file selected")
        self.label_file1.grid(row=1, column=1, columnspan=2, sticky="w")

        # Pulsante per il secondo file
        self.btn_file2 = tk.Button(master, text="Load the second model", command=self.load_file2)
        self.btn_file2.grid(row=2, column=0, padx=5)
        self.label_file2 = tk.Label(master, text="No file selected")
        self.label_file2.grid(row=2, column=1, columnspan=2, sticky="w")

        # Pulsante di elaborazione
        self.btn_process = tk.Button(master, text="Process the models", command=self.process_files)
        self.btn_process.grid(row=3, column=0, columnspan=3, pady=10)

        # Pulsante per aprire l'output
        self.btn_open = tk.Button(master, text="Apri Output", command=self.open_output, state=tk.DISABLED)
        self.btn_open.grid(row=4, column=0, columnspan=3, pady=5)

        # Status label
        self.status_label = tk.Label(master, text="")
        self.status_label.grid(row=5, column=0, columnspan=3)

    def load_file1(self):
        self.file1_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        self.label_file1.config(text=self.file1_path.split("/")[-1])

    def load_file2(self):
        self.file2_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        self.label_file2.config(text=self.file2_path.split("/")[-1])

    def process_files(self):
        if not self.file1_path or not self.file2_path:
            messagebox.showerror("Error", "Select two PLY files to process! ")
            return

        try:
            self.status_label.config(text="Loading...")
            self.master.update()

            # Elaborazione dei file
            handler1 = SplatPLYHandler()
            handler2 = SplatPLYHandler()

            handler1.load_ply(self.file1_path)
            handler2.load_ply(self.file2_path)

            transformation = handler2.align_icp(handler1)
            handler2.apply_transformation(matrix4d=transformation)
            merged_handler = handler1.compare_and_merge(handler2, mode=1)
            
            # Crea la cartella output se non esiste
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            merged_handler.save_ply(self.output_path)

            self.btn_open.config(state=tk.NORMAL)
            self.status_label.config(text="Elaborazione completata!")
            
        except Exception as e:
            messagebox.showerror("Errore", str(e))
            self.status_label.config(text="Errore durante l'elaborazione")

    def open_output(self):
        if os.path.exists(self.output_path):
            try:
                os.startfile(self.output_path)  # Per Windows
            except:
                import subprocess
                subprocess.call(("open", self.output_path))  # Per macOS
                # Per Linux usare 'xdg-open'
        else:
            messagebox.showerror("Errore", "File output non trovato!")

if __name__ == "__main__":

    ''' ### No more used ###
    source_path = "../data/input/SatiroEBaccante_broken2.ply"
    target_path = "../data/input/SatiroEBaccante_broken.ply"
    voxel_parameter = 0.01 

    # Loading the first PLY file
    ply1_points, ply1_colors = load_plydata(source_path)

    # Loading the second PLY file
    ply2_points, ply2_colors = load_plydata(target_path)

    # Applica la trasformazione al secondo modello
    result_scale, result_trl, result_rot = compute_transformation(source_path, target_path, voxel_parameter)
    ply2_points = apply_transformation(ply2_points, result_scale, result_trl, result_rot)

    # Confronta i punti e cambia colore
    ply1_colors, ply2_colors = compare_and_color(ply1_points, ply2_points, ply1_colors, ply2_colors)

    # Unisci i punti e i colori dei due modelli
    merged_points = np.vstack([ply1_points, ply2_points])
    merged_colors = np.vstack([ply1_colors, ply2_colors])

    # Crea un nuovo file PLY
    merged_vertices = np.zeros(merged_points.shape[0], dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    merged_vertices['x'] = merged_points[:, 0]
    merged_vertices['y'] = merged_points[:, 1]
    merged_vertices['z'] = merged_points[:, 2]
    merged_vertices['red'] = merged_colors[:, 0]
    merged_vertices['green'] = merged_colors[:, 1]
    merged_vertices['blue'] = merged_colors[:, 2]

    # Salva il file PLY risultante
    merged_ply = PlyData([PlyElement.describe(merged_vertices, 'vertex')], text=True)
    merged_ply.write("../data/output/merged_model.ply")

    pcd = load_ply_as_point_cloud("../data/output/merged_model.ply")
    o3d.visualization.draw_geometries([pcd], window_name="Merged Model")
    '''

    source_path = "../data/input/SatiroEBaccante_broken2.ply"
    target_path = "../data/input/SatiroEBaccante_broken.ply"

    root = tk.Tk()
    gui = PLYProcessorGUI(root)
    root.mainloop()