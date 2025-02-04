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
from tkinter import ttk,filedialog, messagebox, colorchooser
import numpy as np
import os
import time  
from PIL import Image, ImageTk

class CreateToolTip(object):
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text='widget info'):
        self.waittime = 500     # milliseconds
        self.wraplength = 300   # pixels
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.showtip)

    def unschedule(self):
        id_ = self.id
        self.id = None
        if id_:
            self.widget.after_cancel(id_)

    def showtip(self, event=None):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)  # rimuove bordi e title bar
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background='white', relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        if self.tw:
            self.tw.destroy()
        self.tw = None

class PLYProcessorGUI:
    def __init__(self, master):
        self.master = master
        master.title("PLY Model Processor")
        
        # Variabili per i percorsi dei file
        self.file1_path = ""
        self.file2_path = ""
        self.output_dir = os.path.abspath("../data/output")
        self.output_path = os.path.join(self.output_dir, "merged_model.ply")

        # Configurazione stile
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TLabel", padding=5, font=("Arial", 10))
        self.style.configure("TFrame", background="#f0f0f0")

        # Frame principale
        self.main_frame = ttk.Frame(master, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Aggiungi un'immagine (logo)
        self.add_logo()

        # Widget GUI
        self.label = tk.Label(master)
        self.label.grid(row=0, column=0, columnspan=3, pady=10)

        # Pulsante per il primo file
        self.btn_file1 = ttk.Button(self.main_frame, text="Load Model 1", command=self.load_file1)
        self.btn_file1.grid(row=1, column=0, padx=5, pady=5)
        self.label_file1 = ttk.Label(self.main_frame, text="No file selected", foreground="#666")
        self.label_file1.grid(row=1, column=1, columnspan=2, sticky="w")

        # Pulsante per il secondo file
        self.btn_file2 = ttk.Button(self.main_frame, text="Load Model 2", command=self.load_file2)
        self.btn_file2.grid(row=2, column=0, padx=5, pady=5)
        self.label_file2 = ttk.Label(self.main_frame, text="No file selected", foreground="#666")
        self.label_file2.grid(row=2, column=1, columnspan=2, sticky="w")

        # Pulsante per selezionare la cartella di output
        self.btn_output = ttk.Button(self.main_frame, text="Select Output Folder", command=self.select_output_folder)
        self.btn_output.grid(row=3, column=0, padx=5, pady=5)

        # Troncatura del path di output
        short_output_dir = os.path.basename(self.output_dir) if self.output_dir else "Not selected"
        self.label_output = ttk.Label(self.main_frame, text=f"Output folder: {short_output_dir}", foreground="#666")
        self.label_output.grid(row=3, column=1, columnspan=2, sticky="w")

        # Sezione per la selezione dei parametri (threshold, rgb, alpha e mode)
        self.params_frame = ttk.LabelFrame(self.main_frame, text="Parameters", padding="10")
        self.params_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")

        # Aggiunge un'icona "?" per il tooltip
        self.params_info_label = ttk.Label(self.params_frame, text="?", foreground="blue", cursor="question_arrow")
        self.params_info_label.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)
        tooltip_text = (
            "Parameters:\n"
            "- Threshold: Distance threshold for considering points as mismatched.\n"
            "- RGB1: RGB color for missing/removed parts (default: [1.0, 0.0, 0.0]).\n"
            "- RGB2: RGB color for extra/added parts (default: [0.0, 1.0, 0.0]).\n"
            "- Alpha1: Alpha for transparency of mismatched points in the first PLY (default: nan).\n"
            "- Alpha2: Alpha for transparency of mismatched points in the second PLY (default: nan).\n"
            "- Mode: Determines how to apply color changes (default: 0)."
        )
        CreateToolTip(self.params_info_label, text=tooltip_text)
        
        # Threshold
        ttk.Label(self.params_frame, text="Threshold:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_threshold = ttk.Entry(self.params_frame, width=10)
        self.entry_threshold.insert(0, "0.001")
        self.entry_threshold.grid(row=0, column=1, padx=5, pady=2)
        
        # RGB 1: Sostituisce le entry numeriche con un pulsante per aprire il color chooser
        ttk.Label(self.params_frame, text="RGB 1:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.btn_rgb1 = ttk.Button(self.params_frame, text="Choose Color", command=self.choose_rgb1)
        self.btn_rgb1.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.rgb1_color_display = tk.Label(self.params_frame, text="#ff0000", background="#ff0000", width=10)
        self.rgb1_color_display.grid(row=1, column=2, padx=5, pady=2, sticky="w")
        
        # RGB 2: Simile a RGB 1, default verde (#00ff00)
        ttk.Label(self.params_frame, text="RGB 2:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.btn_rgb2 = ttk.Button(self.params_frame, text="Choose Color", command=self.choose_rgb2)
        self.btn_rgb2.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.rgb2_color_display = tk.Label(self.params_frame, text="#00ff00", background="#00ff00", width=10)
        self.rgb2_color_display.grid(row=2, column=2, padx=5, pady=2, sticky="w")
        
        # Alpha values
        ttk.Label(self.params_frame, text="Alpha 1:").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.entry_alpha1 = ttk.Entry(self.params_frame, width=10)
        self.entry_alpha1.insert(0, "nan")
        self.entry_alpha1.grid(row=3, column=1, padx=5, pady=2)
        ttk.Label(self.params_frame, text="Alpha 2:").grid(row=3, column=2, padx=5, pady=2, sticky="w")
        self.entry_alpha2 = ttk.Entry(self.params_frame, width=10)
        self.entry_alpha2.insert(0, "nan")
        self.entry_alpha2.grid(row=3, column=3, padx=5, pady=2)
        
        # Mode
        ttk.Label(self.params_frame, text="Mode:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.mode_var = tk.IntVar(value=0)  # default mode 0
        ttk.Radiobutton(self.params_frame, text="0", variable=self.mode_var, value=0).grid(row=4, column=1, padx=2, pady=2)
        ttk.Radiobutton(self.params_frame, text="1", variable=self.mode_var, value=1).grid(row=4, column=2, padx=2, pady=2)

        # Pulsante di elaborazione
        self.btn_process = ttk.Button(self.main_frame, text="Process Models", command=self.process_files)
        self.btn_process.grid(row=5, column=0, columnspan=3, pady=10)

        # Pulsante per aprire la cartella di output
        self.btn_open = ttk.Button(self.main_frame, text="Open Output Folder", command=self.open_output_folder, state=tk.DISABLED)
        self.btn_open.grid(row=6, column=0, columnspan=3, pady=5)

        # Barra di progresso
        self.progress = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, pady=10)

        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready", foreground="#333")
        self.status_label.grid(row=7, column=0, columnspan=3)

    def add_logo(self):
        try:
            image_path = "C:/Users/bianc/Downloads/logo_unige.png"  
            original_image = Image.open(image_path)
            
            # Ridimensiona l'immagine (esempio: larghezza 200px, altezza proporzionale)
            width, height = original_image.size
            new_width = 200
            new_height = int((new_width / width) * height)
            resized_image = original_image.resize((new_width, new_height), Image.ANTIALIAS)
            
            # Converte l'immagine in un formato compatibile con Tkinter
            self.logo_image = ImageTk.PhotoImage(resized_image)
            
            # Crea un widget Label per visualizzare l'immagine
            logo_label = tk.Label(self.main_frame, image=self.logo_image, bg="#f0f0f0")
            logo_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))  # Posiziona sopra gli altri elementi
        except Exception as e:
            print(f"Errore durante il caricamento del logo: {e}")    

    def load_file1(self):
        self.file1_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        self.label_file1.config(text=self.file1_path.split("/")[-1])

    def load_file2(self):
        self.file2_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        self.label_file2.config(text=self.file2_path.split("/")[-1])

    def choose_rgb1(self):
        # Apre il color chooser con colore iniziale rosso
        rgb_tuple, hex_color = colorchooser.askcolor(initialcolor="red", title="Choose RGB 1 Color")
        if rgb_tuple:
            # Converte i valori da 0-255 a 0-1
            self.rgb1 = [v / 255.0 for v in rgb_tuple]
            # Aggiorna il display del colore
            self.rgb1_color_display.configure(text=hex_color, background=hex_color)

    def choose_rgb2(self):
        # Apre il color chooser con colore iniziale verde
        rgb_tuple, hex_color = colorchooser.askcolor(initialcolor="#green", title="Choose RGB 2 Color")
        if rgb_tuple:
            self.rgb2 = [v / 255.0 for v in rgb_tuple]
            self.rgb2_color_display.configure(text=hex_color, background=hex_color)

    def save_report(self, handler1, handler2, diff_points, processing_time):
        report_dir = os.path.join(os.path.dirname(self.output_path), "reports")
        os.makedirs(report_dir, exist_ok=True)

        report_filename = os.path.splitext(os.path.basename(self.output_path))[0] + "_report.txt"
        report_path = os.path.join(report_dir, report_filename)

        avg_distance = np.mean(diff_points) if len(diff_points) > 0 else 0
        max_distance = np.max(diff_points) if len(diff_points) > 0 else 0
        num_points_model1 = handler1.elements["vertex"]["count"]
        num_points_model2 = handler2.elements["vertex"]["count"]

        with open(report_path, "w") as f:
            f.write(f"Report per il modello: {os.path.basename(self.output_path)}\n")
            f.write(f"Distanza media tra i punti diversi: {avg_distance:.4f}\n")
            f.write(f"Distanza massima tra i punti diversi: {max_distance:.4f}\n")
            f.write(f"Numero totale di punti in ${self.file1_path}: {num_points_model1}\n")
            f.write(f"Numero totale di punti in ${self.file2_path}: {num_points_model2}\n")
            f.write(f"Tempo di elaborazione: {processing_time:.2f} secondi\n")

    def process_files(self):
        if not self.file1_path or not self.file2_path:
            messagebox.showerror("Error", "Please select both PLY files!")
            return

        try:
            start_time = time.time()    

            self.status_label.config(text="Processing... Please wait")
            self.progress["value"] = 0
            self.master.update()

            # Elaborazione dei file
            handler1 = SplatPLYHandler()
            handler2 = SplatPLYHandler()
            self.progress["value"] = 25
            self.master.update()

           # Load files (25% progress)
            handler1.load_ply(self.file1_path)
            handler2.load_ply(self.file2_path)
            self.progress["value"] = 25
            self.master.update()

            # Calculate alignment (50% progress)
            transformation = handler2.align_icp(handler1)
            self.progress["value"] = 50
            self.master.update()

            # Apply transformation (75% progress)
            handler2.apply_transformation(matrix4d=transformation)
            self.progress["value"] = 75
            self.master.update()

            # Leggi i parametri inseriti nell'interfaccia
            try:
                threshold = float(self.entry_threshold.get())
            except ValueError:
                threshold = 0.001  # default in caso di errore

            # Usa i colori scelti (self.rgb1 e self.rgb2)
            rgb1 = self.rgb1
            rgb2 = self.rgb2

            alpha1_str = self.entry_alpha1.get().strip()
            alpha2_str = self.entry_alpha2.get().strip()
            alpha1 = np.nan if alpha1_str.lower() == "nan" or alpha1_str=="" else float(alpha1_str)
            alpha2 = np.nan if alpha2_str.lower() == "nan" or alpha2_str=="" else float(alpha2_str)

            mode = self.mode_var.get()

            # Merge and save (100% progress)
            merged_handler = handler1.compare_and_merge(
                handler2,
                threshold=threshold,
                rgb1=rgb1,
                rgb2=rgb2,
                alpha1=alpha1,
                alpha2=alpha2,
                mode=mode
            )
            os.makedirs(self.output_dir, exist_ok=True)
            merged_handler.save_ply(self.output_path)
            self.progress["value"] = 100

            # Calcola le differenze tra i punti
            diff_points = handler1.compare_points(handler2, threshold=0.01)

            # Misura il tempo di elaborazione
            processing_time = time.time() - start_time

            # Save report
            self.save_report(handler1, handler2, diff_points, processing_time)

            self.btn_open.config(state=tk.NORMAL)
            self.status_label.config(text="Processing complete!", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_label.config(text="Processing failed", foreground="red")
            self.progress["value"] = 0

    def open_output_folder(self):
        if os.path.exists(self.output_dir):
            try:
                os.startfile(self.output_dir)  # Windows
            except:
                try:
                    import subprocess
                    subprocess.call(["open", self.output_dir])  # macOS
                except:
                    subprocess.call(["xdg-open", self.output_dir])  # Linux
        else:
            messagebox.showerror("Error", "Output folder not found!")

    def select_output_folder(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            short_output_dir = os.path.basename(self.output_dir)
            self.label_output.config(text=f"Output folder: {short_output_dir}")
            self.output_path = os.path.join(self.output_dir, "merged_model.ply")

    
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

    # source_path = "../data/input/SatiroEBaccante_broken2.ply"
    # target_path = "../data/input/SatiroEBaccante_broken.ply"

    root = tk.Tk()
    gui = PLYProcessorGUI(root)
    root.mainloop()