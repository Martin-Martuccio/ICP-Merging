from SplatPLYHandler import SplatPLYHandler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
import os
import sys
import subprocess
import time
import webbrowser
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
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None

    def showtip(self, event=None):
        if not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)  # Remove border and title bar
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                         background='white', relief='solid', borderwidth=1,
                         wraplength=self.wraplength)
        label.pack(ipadx=1)

    def hidetip(self):
        if self.tw:
            self.tw.destroy()
        self.tw = None

    def update_text(self, new_text):
        self.text = new_text

class PLYProcessorGUI:
    def __init__(self, master):
        self.max_length = 15  # Maximum length of visible text in the file selection labels
        self.master = master
        master.title("PLY Model Processor")
        
        # Variables for file processing
        self.file1_path = ""
        self.file2_path = ""
        self.output_dir = os.path.abspath("../data/output")
        self.output_path = os.path.join(self.output_dir, "merged_model.ply")

        # Configuration style
        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", background="#ccc")
        self.style.configure("TLabel", padding=5, font=("Arial", 10))
        self.style.configure("TFrame", background="#f0f0f0")

        # Main Frame
        self.main_frame = ttk.Frame(master, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Add UniGe logo
        self.add_logo("../images/logo_unige.png")

        # Widget GUI
        self.label = tk.Label(master)
        self.label.grid(row=0, column=0, columnspan=3, pady=10)

        # First File Button
        self.btn_file1 = ttk.Button(self.main_frame, text="Load Model 1", command=self.load_file1)
        self.btn_file1.grid(row=1, column=0, padx=5, pady=5)
        self.label_file1 = ttk.Label(self.main_frame, text="No file selected", width=self.max_length, foreground="#666")
        self.label_file1.grid(row=1, column=1, columnspan=2, sticky="w")

        # Second File Button
        self.btn_file2 = ttk.Button(self.main_frame, text="Load Model 2", command=self.load_file2)
        self.btn_file2.grid(row=2, column=0, padx=5, pady=5)
        self.label_file2 = ttk.Label(self.main_frame, text="No file selected", width=self.max_length, foreground="#666")
        self.label_file2.grid(row=2, column=1, columnspan=2, sticky="w")

        # Output Directory Button
        self.btn_output = ttk.Button(self.main_frame, text="Select Output Folder", command=self.select_output_folder)
        self.btn_output.grid(row=3, column=0, padx=5, pady=5)

        # Output path truncation
        short_output_dir = os.path.basename(self.output_dir) if self.output_dir else "Not selected"
        self.label_output = ttk.Label(self.main_frame, text=short_output_dir, width=self.max_length, foreground="#666")
        self.label_output.grid(row=3, column=1, columnspan=2, sticky="w")
        if self.output_dir:
            self.label_output.tooltip = CreateToolTip(self.label_output, self.output_dir)

        # Section for parameters selection (threshold, rgb, alpha e mode)
        self.params_frame = ttk.LabelFrame(self.main_frame, text="Parameters", padding="10")
        self.params_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky="ew")

        # Add "?" icon for tooltip
        self.params_info_label = ttk.Label(self.params_frame, text="?", foreground="blue", cursor="question_arrow")
        self.params_info_label.place(relx=1.0, rely=0.0, anchor="ne", x=-5, y=5)
        tooltip_text = (
            "Parameters:\n"
            "- Threshold: Distance threshold for considering points as mismatched.\n"
            "- RGB1: RGB color for missing/removed parts (default: [1.0, 0.0, 0.0]).\n"
            "- RGB2: RGB color for extra/added parts (default: [0.0, 1.0, 0.0]).\n"
            "- Alpha1: Alpha for transparency of mismatched points in the first PLY (default: nan). [0 <= Alpha1 <= 1 ]\n"
            "- Alpha2: Alpha for transparency of mismatched points in the second PLY (default: nan). [0 <= Alpha2 <= 1 ]\n"
            "- Mode: Determines how to apply color changes (default: 0). [0: Without shadows; 1: With shadows]"
        )
        CreateToolTip(self.params_info_label, text=tooltip_text)
        
        # Threshold
        ttk.Label(self.params_frame, text="Threshold:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.entry_threshold = ttk.Entry(self.params_frame, width=10)
        self.entry_threshold.insert(0, "0.001")
        self.entry_threshold.grid(row=0, column=1, padx=5, pady=2)
        
        # RGB 1: Replaces numeric entries with a button to open the color chooser, default red (#ff0000)
        ttk.Label(self.params_frame, text="RGB 1:").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.btn_rgb1 = ttk.Button(self.params_frame, text="Choose Color", command=self.choose_rgb1)
        self.btn_rgb1.grid(row=1, column=1, padx=5, pady=2, sticky="w")
        self.rgb1 = [1.0, 0.0, 0.0]
        self.rgb1_color_display = tk.Label(self.params_frame, text="#ff0000", background="#ff0000", width=10)
        self.rgb1_color_display.grid(row=1, column=2, padx=5, pady=2, sticky="w")
        
        # RGB 2: Same as RGB 1, default green (#00ff00)
        ttk.Label(self.params_frame, text="RGB 2:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.btn_rgb2 = ttk.Button(self.params_frame, text="Choose Color", command=self.choose_rgb2)
        self.btn_rgb2.grid(row=2, column=1, padx=5, pady=2, sticky="w")
        self.rgb2 = [0.0, 1.0, 0.0]
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

        # Elaboration Button
        self.btn_process = ttk.Button(self.main_frame, text="Merge Models", command=self.process_files)
        self.btn_process.grid(row=5, column=0, columnspan=1, pady=10)

        # Button to open the GitHub page of the Custom Viewer
        self.btn_viewer = ttk.Button(self.main_frame, text="Open Viewer", command=self.viewer_link)
        self.btn_viewer.grid(row=5, column=1, columnspan=2, sticky="w")

        # Button to open the output folder
        self.btn_open = ttk.Button(self.main_frame, text="Open Output Folder", command=self.open_output_folder, state=tk.DISABLED)
        self.btn_open.grid(row=6, column=0, columnspan=3, pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(self.main_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=8, column=0, columnspan=3, pady=10)

        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready", foreground="#333")
        self.status_label.grid(row=7, column=0, columnspan=3)
    
    def viewer_link(self):
        url = "https://biaperass.github.io/Gaussian-Splatting-WebGL/"
        try:
            if sys.platform.startswith("win"):  # Windows
                webbrowser.open(url)
            elif sys.platform.startswith("darwin"):  # macOS
                webbrowser.open(url)
            elif sys.platform.startswith("linux"):
                if "microsoft" in os.uname().release.lower():
                    try:
                        subprocess.call(["wslview", url])  # WSL with WSLView
                    except FileNotFoundError:
                        subprocess.call(["explorer.exe", url]) # WSL without WSLView
                else:
                    webbrowser.open(url)  # Linux
            else:
                messagebox.showerror("Error", "Unsupported OS")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open the browser: {e}")

    def add_logo(self, image_path):
        try:
            original_image = Image.open(image_path)
            
            # Resize the image (example: 100px width, proportional height)
            width, height = original_image.size
            new_width = 200
            new_height = int((new_width / width) * height)
            resized_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Converts the image to a Tkinter compatible format
            self.logo_image = ImageTk.PhotoImage(resized_image)
            
            # Create a Label widget to display the image
            logo_label = tk.Label(self.main_frame, image=self.logo_image, bg="#f0f0f0")
            logo_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))  # Create a Label widget to display the image
        except Exception as e:
            print(f"Error loading logo: {e}")

    def load_file1(self):
        self.file1_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        if self.file1_path:
            filename = os.path.basename(self.file1_path)
            if len(filename) > self.max_length:
                filename = filename[:self.max_length - 3] + "..."  # Truncates the file name
            self.label_file1.config(text=filename)

            if hasattr(self.label_file1, "tooltip"):
                self.label_file1.tooltip.update_text(self.file1_path)
            else:
                self.label_file1.tooltip = CreateToolTip(self.label_file1, self.file1_path)

    def load_file2(self):
        self.file2_path = filedialog.askopenfilename(filetypes=[("PLY files", "*.ply")])
        if self.file2_path:
            filename = os.path.basename(self.file2_path)
            if len(filename) > self.max_length:
                filename = filename[:self.max_length - 3] + "..."  # Truncates the file name
            self.label_file2.config(text=filename)

            if hasattr(self.label_file2, "tooltip"):
                self.label_file2.tooltip.update_text(self.file2_path)
            else:
                self.label_file2.tooltip = CreateToolTip(self.label_file2, self.file2_path)

    def select_output_folder(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            output_dir = os.path.basename(self.output_dir)
            if len(output_dir) > self.max_length:
                output_dir = output_dir[:self.max_length - 3] + "..."  # Truncates the file name
            self.label_output.config(text=output_dir)
            self.output_path = os.path.join(self.output_dir, "merged_model.ply")

            if hasattr(self.label_output, "tooltip"):
                self.label_output.tooltip.update_text(self.output_dir)
            else:
                self.label_output.tooltip = CreateToolTip(self.label_output, self.output_dir)

    def choose_rgb1(self):
        # Obtain the initial color
        initial_color = "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in self.rgb1))
        # Opens the color chooser
        rgb_tuple, hex_color = colorchooser.askcolor(initialcolor=initial_color, title="Choose RGB 1 Color")
        if rgb_tuple:
            # Converts values ​​from 0-255 to 0-1
            self.rgb1 = [v / 255.0 for v in rgb_tuple]
            # Update the color display
            self.rgb1_color_display.configure(text=hex_color, background=hex_color)

    def choose_rgb2(self):
        # Obtain the initial color
        initial_color = "#{:02x}{:02x}{:02x}".format(*(int(c * 255) for c in self.rgb2))
        # Opens the color chooser
        rgb_tuple, hex_color = colorchooser.askcolor(initialcolor=initial_color, title="Choose RGB 2 Color")
        if rgb_tuple:
            # Converts values ​​from 0-255 to 0-1
            self.rgb2 = [v / 255.0 for v in rgb_tuple]
            # Update the color display
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
            f.write(f"Report for the model: {os.path.basename(self.output_path)}\n")
            f.write(f"Average distance between different points: {avg_distance:.4f}\n")
            f.write(f"Maximum distance between different points: {max_distance:.4f}\n")
            f.write(f"Total number of points in ${self.file1_path}: {num_points_model1}\n")
            f.write(f"Total number of points in ${self.file2_path}: {num_points_model2}\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")

    def process_files(self):
        if not self.file1_path or not self.file2_path:
            messagebox.showerror("Error", "Please select both PLY files!")
            return

        try:
            start_time = time.time()    

            self.status_label.config(text="Processing... Please wait")
            self.progress["value"] = 0
            self.master.update()

            # Load files (25% progress)
            handler1 = SplatPLYHandler(filepath = self.file1_path)
            handler2 = SplatPLYHandler(filepath = self.file2_path)
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

            # Read the parameters entered in the interface
            try:
                threshold = float(self.entry_threshold.get())
            except ValueError:
                threshold = 0.001  # Default in case of error

            # Use chosen colors (rgb1 and rgb2)
            rgb1 = self.rgb1
            rgb2 = self.rgb2

            # Use chosen alpha
            alpha1_str = self.entry_alpha1.get().strip()
            alpha2_str = self.entry_alpha2.get().strip()
            alpha1 = np.nan if alpha1_str.lower() == "nan" or alpha1_str == "" else float(alpha1_str)
            alpha2 = np.nan if alpha2_str.lower() == "nan" or alpha2_str == "" else float(alpha2_str)

            # Use chosen mode
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
            self.master.update()

            # Calculate the differences between the points
            diff_points = handler1.compare_points(handler2, threshold=0.01)

            # Measure processing time
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
            if sys.platform.startswith("win"):
                os.startfile(self.output_dir)  # Windows
            elif sys.platform.startswith("darwin"):
                subprocess.call(["open", self.output_dir])  # macOS
            elif sys.platform.startswith("linux"):
                if "microsoft" in os.uname().release.lower():
                    subprocess.call(["explorer.exe", self.output_dir.replace("/", "\\")]) # WSL
                else:
                    subprocess.call(["xdg-open", self.output_dir]) # Linux
            else:
                messagebox.showerror("Error", "Unsupported OS")
        else:
            messagebox.showerror("Error", "Output folder not found!")