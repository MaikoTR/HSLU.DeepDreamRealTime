import os
import sys
import time
import threading
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.keras.applications import InceptionV3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
print(f"Devices available: {tf.config.list_physical_devices()}")

class GuidedDeepDream:
    def __init__(self, 
                 model_name='inception_v3',
                 layers=None):
        """
        Initialize the Guided DeepDream model.
        
        Args:
            model_name: Name of the model to use (currently only 'inception_v3' supported)
            layers: List of layer names to use for feature extraction
        """
        self.model_name = model_name

        self.jitter_dx = 0
        self.jitter_dy = 0

        if model_name == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')

            if layers is None:
                self.layers = ['mixed3', 'mixed5', 'mixed7']
            else:
                self.layers = layers

            outputs = [base_model.get_layer(name).output for name in self.layers]
            self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

            self.min_img_size = (299, 299)
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        print(f"Model initialized with layers: {self.layers}")
    
    def preprocess_image(self, img_path, target_size=None):
        """Load and preprocess image for the model"""
        img = PIL.Image.open(img_path).convert("RGB")

        original_size = img.size
 
        if target_size is None:
            target_size = (max(original_size[0], self.min_img_size[0]),
                          max(original_size[1], self.min_img_size[1]))

        img = img.resize(target_size, PIL.Image.LANCZOS)

        img_array = np.array(img)

        self.original_shape = img_array.shape[:-1]

        img_array = np.expand_dims(img_array, axis=0)

        if self.model_name == 'inception_v3':
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            
        return img_array
    
    def deprocess_image(self, img):
        """Convert processed image back to displayable format with improved quality"""
        img = img.copy()
        img = img[0]

        if self.model_name == 'inception_v3':
            img /= 2.0
            img += 0.5

        img *= 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img
    
    def add_jitter(self, image, jitter=32):
        """Add random jitter to image"""
        if jitter > 0:
            self.jitter_dx, self.jitter_dy = np.random.randint(-jitter, jitter+1, 2)
            image = tf.roll(image, self.jitter_dx, axis=1)
            image = tf.roll(image, self.jitter_dy, axis=2)
        return image
    
    def remove_jitter(self, image, jitter=32):
        """Remove random jitter from image"""
        if jitter > 0:
            image = tf.roll(image, -self.jitter_dx, axis=1)
            image = tf.roll(image, -self.jitter_dy, axis=2)
        return image
    
    def calculate_loss(self, dream_features, ref_features, guidance_weight):
        """Calculate the combined loss for guided dreaming"""
        loss = 0

        layer_weights = np.linspace(0.5, 1.0, len(dream_features))
        
        for i, (dream_feat, ref_feat) in enumerate(zip(dream_features, ref_features)):
            standard_loss = tf.reduce_mean(tf.square(dream_feat))

            dream_feat_norm = dream_feat / (tf.math.reduce_std(dream_feat) + 1e-8)
            ref_feat_norm = ref_feat / (tf.math.reduce_std(ref_feat) + 1e-8)
            
            dream_flat = tf.reshape(dream_feat_norm, [tf.shape(dream_feat_norm)[0], -1])
            ref_flat = tf.reshape(ref_feat_norm, [tf.shape(ref_feat_norm)[0], -1])

            dot_product = tf.reduce_sum(dream_flat * ref_flat, axis=1)
            dream_norm = tf.sqrt(tf.reduce_sum(tf.square(dream_flat), axis=1) + 1e-8)
            ref_norm = tf.sqrt(tf.reduce_sum(tf.square(ref_flat), axis=1) + 1e-8)
            
            similarity = tf.reduce_mean(dot_product / (dream_norm * ref_norm + 1e-8))
            guidance_loss = similarity

            layer_weight = layer_weights[i]
            combined_loss = layer_weight * ((1 - guidance_weight) * standard_loss + 
                             guidance_weight * guidance_loss)
            
            loss += combined_loss
            
        return loss
    
    def octave_process(self, 
                    base_img, 
                    ref_imgs,
                    octave_scale=1.3,
                    num_octaves=4,
                    guidance_weight=0.5,
                    iterations_per_octave=20,
                    step_size=0.01,
                    jitter=32,
                    show_progress=True,
                    progress_callback=None):
        """
        Process the image at different scales for better detail.
        
        Args:
            base_img: Base image tensor
            ref_imgs: List of reference image tensors
            octave_scale: Scale factor between octaves
            num_octaves: Number of octaves scales to process
            guidance_weight: How much to influence with reference (0-1)
            iterations_per_octave: Number of optimization steps per octave
            step_size: Size of each gradient step
            jitter: Amount of random jitter to apply
            show_progress: Whether to show progress images
            progress_callback: Callback function to update UI progress
        """
        original_shape = tf.shape(base_img)[1:3]
        shape_float = tf.cast(original_shape, tf.float32)

        num_octaves = min(num_octaves, 5)

        octave_shapes = []
        min_dim = tf.reduce_min(shape_float).numpy()

        while octave_scale ** (num_octaves - 1) > min_dim / 100:
            num_octaves -= 1
            if num_octaves <= 1:
                num_octaves = 1
                break
                
        print(f"Using {num_octaves} octaves based on image size")

        for i in range(num_octaves):
            scale_factor = tf.cast(octave_scale ** (num_octaves - 1 - i), tf.float32)
            new_shape = tf.cast(shape_float / scale_factor, tf.int32)
            new_shape = tf.maximum(new_shape, 100)
            octave_shapes.append(new_shape)

        octave_shape = octave_shapes[-1]
        base_octave = tf.image.resize(base_img, octave_shape, 
                          method=tf.image.ResizeMethod.LANCZOS3)
        detail = tf.zeros_like(base_octave)

        for i, octave_shape in enumerate(reversed(octave_shapes)):
            print(f"Processing octave {i+1}/{num_octaves} - Size: {octave_shape.numpy()}")

            if progress_callback:
                octave_progress = (i / num_octaves) * 100
                progress_callback(octave_progress, f"Processing octave {i+1}/{num_octaves}")

            octave_base = tf.image.resize(base_img, octave_shape, 
                              method=tf.image.ResizeMethod.LANCZOS3)

            if i > 0:
                detail = tf.image.resize(detail, octave_shape, 
                         method=tf.image.ResizeMethod.LANCZOS3)
                detail = tf.reshape(detail, tf.shape(octave_base))

            dream_img = tf.Variable(octave_base + detail)

            resized_ref_imgs = [tf.image.resize(img, octave_shape, 
                               method=tf.image.ResizeMethod.LANCZOS3) 
                               for img in ref_imgs]

            result = self.process_single_octave(
                dream_img, 
                resized_ref_imgs,
                guidance_weight=guidance_weight,
                iterations=iterations_per_octave,
                step_size=step_size * (1 - 0.1 * i),
                jitter=min(jitter, tf.reduce_min(octave_shape).numpy() // 8),
                progress_callback=progress_callback,
                octave_index=i,
                total_octaves=num_octaves
            )

            detail = result - octave_base

        final_result = tf.image.resize(result, original_shape, 
                     method=tf.image.ResizeMethod.LANCZOS5)
            
        return final_result
        
    def process_single_octave(self, 
                             dream_img, 
                             ref_imgs,
                             guidance_weight=0.5,
                             iterations=20,
                             step_size=0.01,
                             jitter=32,
                             progress_callback=None,
                             octave_index=0,
                             total_octaves=1):
        """Process a single octave scale"""
        ref_features_list = []
        for img in ref_imgs:
            ref_features_list.append(self.model(img))

        ref_features = []
        for i in range(len(self.layers)):
            feature_stack = tf.stack([feat_list[i] for feat_list in ref_features_list], axis=0)
            ref_features.append(tf.reduce_mean(feature_stack, axis=0))

        lr_schedule = np.linspace(step_size, step_size * 0.6, iterations)

        dream_var = tf.Variable(dream_img)
        
        for i in range(iterations):
            if progress_callback:
                octave_progress = (octave_index / total_octaves)
                iteration_progress = (i / iterations) / total_octaves
                overall_progress = (octave_progress + iteration_progress) * 100
                
                if i % 5 == 0:
                    progress_callback(
                        overall_progress, 
                        f"Octave {octave_index+1}/{total_octaves}, Step {i+1}/{iterations}"
                    )
            
            current_step_size = lr_schedule[i]
            jittered_img = self.add_jitter(dream_var, jitter)

            with tf.GradientTape() as tape:
                tape.watch(jittered_img)
                dream_features = self.model(jittered_img)

                loss = self.calculate_loss(dream_features, ref_features, guidance_weight)

            gradients = tape.gradient(loss, jittered_img)
            gradients = gradients / (tf.math.reduce_std(gradients) + 1e-8)

            jittered_updated = jittered_img + gradients * current_step_size

            updated_img = self.remove_jitter(jittered_updated, jitter)
            dream_var.assign(tf.clip_by_value(updated_img, -1.0, 1.0))
            
            if i % 5 == 0:
                try:
                    loss_value = float(loss.numpy())
                    print(f"  Step {i+1}/{iterations} - Loss: {loss_value:.4f}")
                except:
                    print(f"  Step {i+1}/{iterations} - Loss value not available")
        
        return dream_var
    
    def guided_deepdream(self, 
                         base_img_path, 
                         reference_img_paths,
                         output_path=None,
                         target_size=None,
                         guidance_weight=0.5,
                         octave_scale=1.3,
                         num_octaves=4,
                         iterations_per_octave=20,
                         step_size=0.01,
                         jitter=32,
                         show_original=False,
                         show_progress=True,
                         high_quality=True,
                         progress_callback=None):
        """
        Enhanced guided DeepDream implementation with improved quality
        
        Args:
            base_img_path: Path to base image to apply effect to
            reference_img_paths: List of paths to reference images
            output_path: Where to save the result (None = don't save)
            target_size: Size to process images at (None = use original size)
            guidance_weight: How much to influence with reference (0-1)
            octave_scale: Scale factor between octaves
            num_octaves: Number of octave scales to process
            iterations_per_octave: Number of optimization steps per octave
            step_size: Size of each gradient step
            jitter: Amount of random jitter to apply
            show_original: Whether to show original images
            show_progress: Whether to show processing progress
            high_quality: Whether to use higher quality resizing
            progress_callback: Callback function to update UI progress
        """
        start_time = time.time()

        original_img = PIL.Image.open(base_img_path).convert("RGB")
        orig_width, orig_height = original_img.size
        original_size = (orig_width, orig_height)
        
        if target_size is None:
            if high_quality:
                process_size = (max(orig_width, self.min_img_size[0]), 
                               max(orig_height, self.min_img_size[1]))
            else:
                process_size = self.min_img_size
        else:
            process_size = target_size
        
        print(f"Processing at size: {process_size}")

        if progress_callback:
            progress_callback(0, "Preprocessing images...")

        base_img = self.preprocess_image(base_img_path, process_size)
        
        ref_imgs = []
        for img_path in reference_img_paths:
            ref_imgs.append(self.preprocess_image(img_path, process_size))

        print(f"Starting guided dreaming with guidance weight: {guidance_weight}")
        print(f"Octaves: {num_octaves}, Scale: {octave_scale}, Iterations per octave: {iterations_per_octave}")
        
        if progress_callback:
            progress_callback(5, "Starting dream process...")
        
        result = self.octave_process(
            base_img, 
            ref_imgs,
            octave_scale=octave_scale,
            num_octaves=num_octaves,
            guidance_weight=guidance_weight,
            iterations_per_octave=iterations_per_octave,
            step_size=step_size,
            jitter=jitter,
            show_progress=show_progress,
            progress_callback=progress_callback
        )
        
        if progress_callback:
            progress_callback(95, "Finalizing result...")

        result_img = self.deprocess_image(result.numpy())

        if high_quality and result_img.shape[:2] != (orig_height, orig_width):
            print(f"Resizing final result to original dimensions: {original_size}")
            result_pil = PIL.Image.fromarray(result_img)
            result_pil = result_pil.resize(original_size, PIL.Image.LANCZOS)
            result_img = np.array(result_pil)

        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            PIL.Image.fromarray(result_img).save(output_path)
            print(f"Result saved to {output_path}")
        
        if progress_callback:
            progress_callback(100, "Dream complete!")
        
        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")
        
        return result_img


class DeepDreamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Guided DeepDream Creator")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        self.base_img_path = None
        self.ref_img_path = None
        self.result_img = None
        self.is_processing = False
        self.model_ready = False

        self.status_text = tk.StringVar()
        self.status_text.set("Initializing...")

        self.create_ui()
        self.init_model()
    
    def init_model(self):
        self.status_text.set("Initializing model...")
        
        def load_model():
            try:
                default_layers = ['mixed3', 'mixed5', 'mixed7']
                self.dreamer = GuidedDeepDream(layers=default_layers)
                self.model_ready = True
                self.root.after(0, lambda: self.status_text.set("Model ready! Select images to begin."))
                self.root.after(0, self.check_ready_state)
            except Exception as e:
                self.root.after(0, lambda: self.status_text.set(f"Error loading model: {str(e)}"))
        threading.Thread(target=load_model, daemon=True).start()
    
    def create_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create split view
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create top section for image selection
        img_select_frame = ttk.LabelFrame(left_frame, text="Image Selection", padding=10)
        img_select_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Base image selection
        ttk.Label(img_select_frame, text="Base Image:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.base_img_label = ttk.Label(img_select_frame, text="No image selected")
        self.base_img_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(img_select_frame, text="Browse...", command=self.select_base_image).grid(row=0, column=2, padx=5, pady=5)
        
        # Reference image selection
        ttk.Label(img_select_frame, text="Reference Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ref_img_label = ttk.Label(img_select_frame, text="No image selected")
        self.ref_img_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(img_select_frame, text="Browse...", command=self.select_ref_image).grid(row=1, column=2, padx=5, pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(left_frame, text="DeepDream Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Model layers
        ttk.Label(params_frame, text="Model Layers:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.layers_var = tk.StringVar(value="mixed3, mixed5, mixed7")
        ttk.Entry(params_frame, textvariable=self.layers_var, width=30).grid(row=0, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Guidance weight
        ttk.Label(params_frame, text="Guidance Weight:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.guidance_weight_var = tk.DoubleVar(value=0.5)
        ttk.Scale(params_frame, variable=self.guidance_weight_var, from_=0.0, to=1.0, length=200).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(params_frame, textvariable=self.guidance_weight_var).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Octaves
        ttk.Label(params_frame, text="Number of Octaves:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.octaves_var = tk.IntVar(value=3)
        ttk.Spinbox(params_frame, from_=1, to=5, textvariable=self.octaves_var, width=5).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Octave scale
        ttk.Label(params_frame, text="Octave Scale:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.octave_scale_var = tk.DoubleVar(value=1.3)
        ttk.Spinbox(params_frame, from_=1.1, to=2.0, increment=0.1, textvariable=self.octave_scale_var, width=5).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Iterations per octave
        ttk.Label(params_frame, text="Iterations per Octave:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.iterations_var = tk.IntVar(value=20)
        ttk.Spinbox(params_frame, from_=10, to=200, increment=10, textvariable=self.iterations_var, width=5).grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Step size
        ttk.Label(params_frame, text="Step Size:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.step_size_var = tk.DoubleVar(value=0.01)
        ttk.Spinbox(params_frame, from_=0.001, to=0.1, increment=0.005, textvariable=self.step_size_var, width=5).grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Jitter
        ttk.Label(params_frame, text="Jitter:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.jitter_var = tk.IntVar(value=32)
        ttk.Spinbox(params_frame, from_=0, to=128, increment=8, textvariable=self.jitter_var, width=5).grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)
        
        # High quality
        self.high_quality_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="High Quality Processing", variable=self.high_quality_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start DeepDream", command=self.start_process)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = ttk.Button(button_frame, text="Save Result", command=self.save_result, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(left_frame)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Label(progress_frame, textvariable=self.status_text).pack(side=tk.LEFT, padx=5)
        
        # Create image display area
        self.image_notebook = ttk.Notebook(right_frame)
        self.image_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Base image tab
        self.base_img_frame = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.base_img_frame, text="Base Image")
        
        self.base_img_canvas = tk.Canvas(self.base_img_frame, bg="#222222")
        self.base_img_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Reference image tab
        self.ref_img_frame = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.ref_img_frame, text="Reference Image")
        
        self.ref_img_canvas = tk.Canvas(self.ref_img_frame, bg="#222222")
        self.ref_img_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Result image tab
        self.result_img_frame = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.result_img_frame, text="Result")
        
        self.result_img_canvas = tk.Canvas(self.result_img_frame, bg="#222222")
        self.result_img_canvas.pack(fill=tk.BOTH, expand=True)
        
    def select_base_image(self):
        """Open a file dialog to select the base image"""
        path = filedialog.askopenfilename(
            title="Select Base Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if path:
            self.base_img_path = path
            self.base_img_label.config(text=os.path.basename(path))
            self.load_and_display_image(path, self.base_img_canvas, "Base Image")
            self.check_ready_state()
    
    def select_ref_image(self):
        """Open a file dialog to select the reference image"""
        path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if path:
            self.ref_img_path = path
            self.ref_img_label.config(text=os.path.basename(path))
            self.load_and_display_image(path, self.ref_img_canvas, "Reference Image")
            self.check_ready_state()
    
    def load_and_display_image(self, path, canvas, label="Image"):
        """Load and display an image in the specified canvas"""
        try:
            canvas.delete("all")
            img = Image.open(path).convert("RGB")

            self.root.update_idletasks()
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
            canvas.create_image(x, y, anchor=tk.NW, image=photo_img)

            info_text = f"{img.width}x{img.height} px"
            canvas.create_text(10, 10, anchor=tk.NW, text=info_text, fill="white", 
                              font=('Arial', 10))
            
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def check_ready_state(self):
        """Check if all required inputs are ready to start processing"""
        if self.model_ready and self.base_img_path and self.ref_img_path:
            self.start_button.config(state=tk.NORMAL)
            self.status_text.set("Ready to process!")
        else:
            self.start_button.config(state=tk.DISABLED)
    
    def update_progress(self, progress_value, status_message):
        """Update the progress bar and status message"""
        self.progress_var.set(progress_value)
        self.status_text.set(status_message)

        if progress_value >= 100:
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
    
    def start_process(self):
        """Start the DeepDream process"""
        if self.is_processing:
            messagebox.showinfo("Processing", "Already processing an image. Please wait.")
            return
        
        if not self.base_img_path or not self.ref_img_path:
            messagebox.showwarning("Missing Images", "Please select both base and reference images.")
            return

        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.status_text.set("Starting process...")

        try:
            layers_text = self.layers_var.get().strip()
            layers = [layer.strip() for layer in layers_text.split(',')]
            
            guidance_weight = self.guidance_weight_var.get()
            num_octaves = self.octaves_var.get()
            octave_scale = self.octave_scale_var.get()
            iterations = self.iterations_var.get()
            step_size = self.step_size_var.get()
            jitter = self.jitter_var.get()
            high_quality = self.high_quality_var.get()

            threading.Thread(
                target=self.run_deep_dream_process,
                args=(layers, guidance_weight, num_octaves, octave_scale, 
                      iterations, step_size, jitter, high_quality),
                daemon=True
            ).start()
            
        except Exception as e:
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Failed to start process: {str(e)}")
    
    def run_deep_dream_process(self, layers, guidance_weight, num_octaves, 
                               octave_scale, iterations, step_size, jitter, high_quality):
        """Run the DeepDream process in a background thread"""
        try:
            self.root.after(0, lambda: self.update_progress(1, "Initializing model with selected layers..."))

            try:
                # Validate layers to ensure they exist in the model
                self.dreamer = GuidedDeepDream(layers=layers)
            except ValueError as e:
                # Handle specific layer errors
                if "not found in model" in str(e):
                    self.root.after(0, lambda: self.handle_error(
                        f"Invalid layer name. Please use valid InceptionV3 layers such as: "
                        f"mixed0, mixed1, mixed2, mixed3, mixed4, mixed5, mixed6, mixed7, mixed8, mixed9, mixed10"
                    ))
                    return
                else:
                    raise
                    
            self.root.after(0, lambda: self.update_progress(2, "Setting up processing..."))

            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_output_path = os.path.join(temp_dir, f"deepdream_result_{int(time.time())}.jpg")

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
                show_original=False,
                show_progress=False,
                high_quality=high_quality,
                progress_callback=lambda progress, status: 
                    self.root.after(0, lambda: self.update_progress(progress, status))
            )

            self.result_img = result_img
            self.result_img_path = temp_output_path

            self.root.after(0, lambda: self.display_result(temp_output_path))
            
        except Exception as e:
            import traceback
            print("DeepDream process error:")
            traceback.print_exc()
            self.root.after(0, lambda: self.handle_error(str(e)))
    
    def display_result(self, img_path):
        """Display the result image in the canvas"""
        self.load_and_display_image(img_path, self.result_img_canvas, "Result")
        self.image_notebook.select(self.result_img_frame)
        self.status_text.set("DeepDream complete!")
        self.start_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
    
    def save_result(self):
        """Save the result image to a user-specified location"""
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
        """Handle errors that occur during processing"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.status_text.set(f"Error: {error_message}")
        messagebox.showerror("Processing Error", error_message)


    def handle_error(self, error_message):
        """Handle errors that occur during processing"""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.status_text.set(f"Error: {error_message}")
        messagebox.showerror("Processing Error", error_message)
        
    def on_window_resize(self, event=None):
        """Handle window resize events to adjust image display"""
        if hasattr(self, '_resize_job'):
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self.refresh_image_displays)
    
    def refresh_image_displays(self):
        """Refresh all image displays after window resize"""
        if self.base_img_path:
            self.load_and_display_image(self.base_img_path, self.base_img_canvas, "Base Image")
        if self.ref_img_path:
            self.load_and_display_image(self.ref_img_path, self.ref_img_canvas, "Reference Image")
        if hasattr(self, 'result_img_path') and self.result_img_path:
            self.load_and_display_image(self.result_img_path, self.result_img_canvas, "Result")


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    os.makedirs("deepdream_results", exist_ok=True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
    
    try:
        root = tk.Tk()
        app = DeepDreamApp(root)

        root.bind("<Configure>", app.on_window_resize)

        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'+{x}+{y}')

        root.mainloop()
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