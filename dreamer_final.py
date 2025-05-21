import os
import sys
import time
import threading
import tempfile
import numpy as np
import tensorflow as tf
import PIL.Image
from tensorflow.keras.applications import InceptionV3
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from utils.guided_deepdream_tiledgradients import ImprovedGuidedDeepDream

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Devices available: {tf.config.list_physical_devices()}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory setup failed: {e}")

class ModernDeepDreamUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Guided DeepDream Creator")
        self.geometry("1280x800")
        
        self.base_img_path = None
        self.ref_img_path = None
        self.result_img = None
        self.is_processing = False
        self.model_ready = False
        
        self.status_text = ctk.StringVar(value="Initializing...")
        
        self.create_ui()  
        self.init_model()
        
        self.bind("<Configure>", self.on_window_resize)
        
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'+{x}+{y}')
    
    def init_model(self):
        self.status_text.set("Initializing model...")
        
        def load_model():
            try:
                default_layers = ['mixed3', 'mixed5', 'mixed7']
                self.dreamer = ImprovedGuidedDeepDream(model_name='inception_v3', layers=default_layers)
                self.model_ready = True
                self.after(0, lambda: self.status_text.set("Model ready! Select images to begin"))
                self.after(0, self.check_ready_state)
            except Exception as e:
                self.after(0, lambda: self.status_text.set(f"Error loading model: {str(e)}"))
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def create_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        
        left_panel = ctk.CTkFrame(self)
        left_panel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        left_panel.grid_columnconfigure(0, weight=1)
        
        img_select_frame = ctk.CTkFrame(left_panel)
        img_select_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        
        ctk.CTkLabel(img_select_frame, text="Image Selection", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=10, pady=5)
        
        base_img_row = ctk.CTkFrame(img_select_frame)
        base_img_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(base_img_row, text="Base Image:").pack(side="left", padx=(0, 10))
        self.base_img_label = ctk.CTkLabel(base_img_row, text="No image selected")
        self.base_img_label.pack(side="left", fill="x", expand=True)
        
        ctk.CTkButton(base_img_row, text="Browse", command=self.select_base_image,
                      width=80).pack(side="right", padx=5)

        ref_img_row = ctk.CTkFrame(img_select_frame)
        ref_img_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(ref_img_row, text="Reference Image:").pack(side="left", padx=(0, 10))
        self.ref_img_label = ctk.CTkLabel(ref_img_row, text="No image selected")
        self.ref_img_label.pack(side="left", fill="x", expand=True)
        
        ctk.CTkButton(ref_img_row, text="Browse", command=self.select_ref_image,
                      width=80).pack(side="right", padx=5)

        params_frame = ctk.CTkFrame(left_panel)
        params_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(params_frame, text="DeepDream Parameters", font=ctk.CTkFont(size=16, weight="bold")).pack(
            anchor="w", padx=10, pady=5)

        layers_row = ctk.CTkFrame(params_frame)
        layers_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(layers_row, text="Model Layers:").pack(side="left", padx=(0, 10))
        self.layers_var = ctk.StringVar(value="mixed3, mixed5, mixed7")
        ctk.CTkEntry(layers_row, textvariable=self.layers_var).pack(side="left", fill="x", expand=True)

        guidance_row = ctk.CTkFrame(params_frame)
        guidance_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(guidance_row, text="Guidance Weight:").pack(side="left", padx=(0, 10))
        self.guidance_weight_var = ctk.DoubleVar(value=0.5)
        ctk.CTkSlider(guidance_row, from_=0.0, to=1.0, variable=self.guidance_weight_var).pack(
            side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkLabel(guidance_row, textvariable=self.guidance_weight_var, width=40).pack(side="right")

        params_grid = ctk.CTkFrame(params_frame)
        params_grid.pack(fill="x", padx=10, pady=5)
        params_grid.grid_columnconfigure(0, weight=1)
        params_grid.grid_columnconfigure(1, weight=1)

        octaves_row = ctk.CTkFrame(params_grid)
        octaves_row.grid(row=0, column=0, padx=(0, 5), pady=5, sticky="ew")
        
        ctk.CTkLabel(octaves_row, text="Octaves:").pack(side="left", padx=(0, 10))
        self.octaves_var = ctk.StringVar(value="3")
        
        octave_entry = ctk.CTkEntry(octaves_row, textvariable=self.octaves_var, width=60)
        octave_entry.pack(side="left")
        
        step_row = ctk.CTkFrame(params_grid)
        step_row.grid(row=1, column=0, padx=(0, 5), pady=5, sticky="ew")
        
        ctk.CTkLabel(step_row, text="Step Size:").pack(side="left", padx=(0, 10))

        self.step_size_var = ctk.StringVar(value="0.01")
        ctk.CTkEntry(step_row, textvariable=self.step_size_var, width=60).pack(side="left")

        scale_row = ctk.CTkFrame(params_grid)
        scale_row.grid(row=0, column=1, padx=(5, 0), pady=5, sticky="ew")
        
        ctk.CTkLabel(scale_row, text="Octave Scale:").pack(side="left", padx=(0, 10))
        self.octave_scale_var = ctk.StringVar(value="1.3")
        ctk.CTkEntry(scale_row, textvariable=self.octave_scale_var, width=60).pack(side="left")

        jitter_row = ctk.CTkFrame(params_grid)
        jitter_row.grid(row=1, column=1, padx=(5, 0), pady=5, sticky="ew")
        
        ctk.CTkLabel(jitter_row, text="Jitter:").pack(side="left", padx=(0, 10))
        self.jitter_var = ctk.StringVar(value="32")
        ctk.CTkEntry(jitter_row, textvariable=self.jitter_var, width=60).pack(side="left")

        iterations_row = ctk.CTkFrame(params_frame)
        iterations_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(iterations_row, text="Iterations per Octave:").pack(side="left", padx=(0, 10))
        self.iterations_var = ctk.IntVar(value=20)
        ctk.CTkSlider(iterations_row, from_=10, to=250, variable=self.iterations_var).pack(
            side="left", fill="x", expand=True, padx=(0, 10))
        ctk.CTkLabel(iterations_row, textvariable=self.iterations_var, width=40).pack(side="right")

        quality_row = ctk.CTkFrame(params_frame)
        quality_row.pack(fill="x", padx=10, pady=5)
        
        self.high_quality_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(quality_row, text="High Quality Processing", 
                        variable=self.high_quality_var).pack(side="left")

        action_frame = ctk.CTkFrame(left_panel)
        action_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        button_row = ctk.CTkFrame(action_frame)
        button_row.pack(fill="x", padx=10, pady=5)
        
        self.start_button = ctk.CTkButton(button_row, text="Start DeepDream", 
                                          command=self.start_process, state="disabled")
        self.start_button.pack(side="left", padx=(0, 10))
        
        self.save_button = ctk.CTkButton(button_row, text="Save Result", 
                                         command=self.save_result, state="disabled")
        self.save_button.pack(side="left")

        progress_row = ctk.CTkFrame(action_frame)
        progress_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(progress_row, text="Progress:").pack(side="left", padx=(0, 10))
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(progress_row)
        self.progress_bar.set(0)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))

        status_row = ctk.CTkFrame(action_frame)
        status_row.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(status_row, textvariable=self.status_text).pack(anchor="w")

        right_panel = ctk.CTkTabview(self)
        right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.base_tab = right_panel.add("Base Image")
        self.ref_tab = right_panel.add("Reference Image")
        self.result_tab = right_panel.add("Result")

        for tab in [self.base_tab, self.ref_tab, self.result_tab]:
            tab.grid_columnconfigure(0, weight=1)
            tab.grid_rowconfigure(0, weight=1)

        self.base_img_canvas = ctk.CTkCanvas(self.base_tab, bg="#1a1a1a", highlightthickness=0)
        self.base_img_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.ref_img_canvas = ctk.CTkCanvas(self.ref_tab, bg="#1a1a1a", highlightthickness=0)
        self.ref_img_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.result_img_canvas = ctk.CTkCanvas(self.result_tab, bg="#1a1a1a", highlightthickness=0)
        self.result_img_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    
    def select_base_image(self):
        path = filedialog.askopenfilename(
            title="Select Base Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if path:
            self.base_img_path = path
            self.base_img_label.configure(text=os.path.basename(path))
            self.load_and_display_image(path, self.base_img_canvas, "Base Image")
            self.check_ready_state()
    
    def select_ref_image(self):
        path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if path:
            self.ref_img_path = path
            self.ref_img_label.configure(text=os.path.basename(path))
            self.load_and_display_image(path, self.ref_img_canvas, "Reference Image")
            self.check_ready_state()
    
    def load_and_display_image(self, path, canvas, label="Image"):
        try:
            canvas.delete("all")
            img = Image.open(path).convert("RGB")

            self.update_idletasks()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            if canvas_width < 50 or canvas_height < 50:
                canvas_width = 500
                canvas_height = 400

            img_ratio = img.width / img.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)

            if img.width > canvas_width or img.height > canvas_height:
                try:
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                except AttributeError:
                    resized_img = img.resize((new_width, new_height), Image.BICUBIC)
            else:
                resized_img = img

            photo_img = ImageTk.PhotoImage(resized_img)
            setattr(self, f"{label.lower().replace(' ', '_')}_photo", photo_img)

            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor="nw", image=photo_img)

            info_text = f"{img.width}x{img.height} px"
            canvas.create_text(10, 10, anchor="nw", text=info_text, fill="white", 
                              font=('Arial', 10))
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def check_ready_state(self):
        if self.model_ready and self.base_img_path and self.ref_img_path:
            self.start_button.configure(state="normal")
            self.status_text.set("Ready to process!")
        else:
            self.start_button.configure(state="disabled")
    
    def update_progress(self, progress_value, status_message):
        self.progress_var.set(progress_value / 100)
        self.progress_bar.set(progress_value / 100)
        self.status_text.set(status_message)

        if progress_value >= 100:
            self.is_processing = False
            self.start_button.configure(state="normal")
            self.save_button.configure(state="normal")

    def get_valid_int(self, string_var, default_value):
        """Safely convert a StringVar to an integer, with a fallback default value"""
        try:
            value = string_var.get().strip()
            if value:
                return int(value)
            return default_value
        except (ValueError, tk.TclError):
            return default_value
            
    def get_valid_float(self, string_var, default_value):
        """Safely convert a StringVar to a float, with a fallback default value"""
        try:
            value = string_var.get().strip()
            if value:
                return float(value)
            return default_value
        except (ValueError, tk.TclError):
            return default_value
    
    def start_process(self):
        if self.is_processing:
            messagebox.showinfo("Processing", "Already processing an image. Please wait.")
            return
        
        if not self.base_img_path or not self.ref_img_path:
            messagebox.showwarning("Missing Images", "Please select both base and reference images.")
            return

        self.is_processing = True
        self.start_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.progress_var.set(0)
        self.progress_bar.set(0)
        self.status_text.set("Starting process...")

        try:
            layers_text = self.layers_var.get().strip()
            layers = [layer.strip() for layer in layers_text.split(',')]
            
            guidance_weight = self.guidance_weight_var.get()
            
            num_octaves = self.get_valid_int(self.octaves_var, 3)
            octave_scale = self.get_valid_float(self.octave_scale_var, 1.3)
            iterations = self.iterations_var.get()
            step_size = self.get_valid_float(self.step_size_var, 0.01)
            jitter = self.get_valid_int(self.jitter_var, 32)
            high_quality = self.high_quality_var.get()

            threading.Thread(
                target=self.run_deep_dream_process,
                args=(layers, guidance_weight, num_octaves, octave_scale, 
                      iterations, step_size, jitter, high_quality),
                daemon=True
            ).start()
            
        except Exception as e:
            self.is_processing = False
            self.start_button.configure(state="normal")
            messagebox.showerror("Error", f"Failed to start process: {str(e)}")
    
    def run_deep_dream_process(self, layers, guidance_weight, num_octaves, 
                               octave_scale, iterations, step_size, jitter, high_quality):
        try:
            self.after(0, lambda: self.update_progress(1, "Initializing model with selected layers..."))

            try:
                self.dreamer = ImprovedGuidedDeepDream(model_name='inception_v3', layers=layers)
            except ValueError as e:
                if "not found in model" in str(e):
                    error_msg = (f"Invalid layer name. Please use valid InceptionV3 layers such as: "
                                f"mixed0, mixed1, mixed2, mixed3, mixed4, mixed5, mixed6, mixed7, mixed8, mixed9, mixed10")
                    self.after(0, lambda msg=error_msg: self.handle_error(msg))
                    return
                else:
                    raise
                    
            self.after(0, lambda: self.update_progress(2, "Setting up processing..."))

            temp_dir = tempfile.gettempdir()
            temp_output_path = os.path.join(temp_dir, f"deepdream_result_{int(time.time())}.jpg")

            original_print = print
            
            def progress_print(*args, **kwargs):
                message = " ".join(map(str, args))

                progress_map = {
                    "Processing octave 1/": 5,
                    "Starting guided dreaming": 3,
                    "Processing octave 2/": 25,
                    "Processing octave 3/": 50,
                    "Processing octave 4/": 75,
                    "Processing octave 5/": 90,
                    "Processing completed": 99
                }

                step_info = None
                for prefix in progress_map:
                    if prefix in message:
                        progress_value = progress_map[prefix]
                        status = message

                        if "Step " in message and "Step " not in prefix:
                            try:
                                step_parts = message.split("Step ")[1].split("/")
                                current_step = int(step_parts[0])
                                total_steps = int(step_parts[0].split(" ")[0])
                                octave_num = int(message.split("Processing octave ")[1].split("/")[0])

                                base_progress = progress_map[f"Processing octave {octave_num}/"]
                                if octave_num < num_octaves:
                                    next_octave = f"Processing octave {octave_num+1}/"
                                    next_progress = progress_map.get(next_octave, 95)
                                else:
                                    next_progress = 95
                                
                                step_progress = (next_progress - base_progress) * (current_step / iterations)
                                progress_value = base_progress + step_progress
                            except:
                                pass
                        
                        self.after(0, lambda p=progress_value, s=message: self.update_progress(p, s))
                        break

                original_print(*args, **kwargs)

            import builtins
            builtins.print = progress_print
            
            try:
                result_img = self.dreamer.guided_deepdream(
                    base_img_path=self.base_img_path,
                    reference_img_paths=[self.ref_img_path],
                    output_path=temp_output_path,
                    guidance_weight=guidance_weight,
                    octave_scale=octave_scale,
                    num_octaves=num_octaves,
                    iterations_per_octave=iterations,
                    step_size=step_size,
                    jitter=jitter,
                    display_results=False,
                    show_progress=False,
                    high_quality=high_quality
                )
                
                self.result_img = result_img
                self.result_img_path = temp_output_path
                
                self.after(0, lambda: self.update_progress(100, "DeepDream complete!"))
                
                self.after(0, lambda path=temp_output_path: self.display_result(path))
                
            finally:
                builtins.print = original_print
            
        except Exception as exc:
            import traceback
            print("DeepDream process error:")
            traceback.print_exc()
            error_message = str(exc)
            self.after(0, lambda msg=error_message: self.handle_error(msg))
    
    def display_result(self, img_path):
        self.load_and_display_image(img_path, self.result_img_canvas, "Result")
        self.status_text.set("DeepDream complete!")
        self.start_button.configure(state="normal")
        self.save_button.configure(state="normal")
    
    def save_result(self):
        if not hasattr(self, 'result_img_path') or not os.path.exists(self.result_img_path):
            messagebox.showerror("Error", "No result image to save.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save DeepDream Result",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), 
                       ("All files", "*.*")]
        )
        
        if save_path:
            try:
                img = Image.open(self.result_img_path)
                img.save(save_path)
                messagebox.showinfo("Success", f"Image saved to {save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def handle_error(self, error_message):
        self.is_processing = False
        self.start_button.configure(state="normal")
        self.status_text.set(f"Error: {error_message}")
        messagebox.showerror("Processing Error", error_message)
        
    def on_window_resize(self, event=None):
        if hasattr(self, '_resize_job'):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(100, self.refresh_image_displays)
    
    def refresh_image_displays(self):
        if self.base_img_path:
            self.load_and_display_image(self.base_img_path, self.base_img_canvas, "Base Image")
        if self.ref_img_path:
            self.load_and_display_image(self.ref_img_path, self.ref_img_canvas, "Reference Image")
        if hasattr(self, 'result_img_path') and self.result_img_path:
            self.load_and_display_image(self.result_img_path, self.result_img_canvas, "Result")


def main():
    try:
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
        
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        os.makedirs("deepdream_results", exist_ok=True)

        app = ModernDeepDreamUI()
        app.mainloop()
        
    except Exception as e:
        import traceback
        print("Unexpected error:")
        traceback.print_exc()
        try:
            messagebox.showerror("Application Error", 
                                f"An unexpected error occurred: {str(e)}\n\n"
                                "See console for details.")
        except:
            pass


if __name__ == "__main__":
    main()