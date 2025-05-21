import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from tensorflow.keras.applications import InceptionV3
import time
import os

class ImprovedGuidedDeepDream:
    def __init__(self, 
                 model_name='inception_v3',
                 layers=None):
        """
        Initialize the Improved Guided DeepDream model with Google-style octave processing.
        
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
        """
        Add random jitter to image using tf.roll for better artifacts prevention.
        
        Args:
            image: The input image tensor
            jitter: Maximum amount of jitter in pixels
            
        Returns:
            The jittered image
        """
        if jitter > 0:
            self.jitter_dx = tf.random.uniform(shape=[], minval=-jitter, maxval=jitter+1, dtype=tf.int32)
            self.jitter_dy = tf.random.uniform(shape=[], minval=-jitter, maxval=jitter+1, dtype=tf.int32)
            
            image = tf.roll(image, shift=[self.jitter_dx, self.jitter_dy], axis=[1, 2])
        
        return image

    def remove_jitter(self, image, jitter=32):
        """
        Remove previously applied random jitter from image.
        
        Args:
            image: The input image tensor
            jitter: Maximum amount of jitter in pixels (unused but kept for API consistency)
            
        Returns:
            The unjittered image
        """
        if jitter > 0:
            image = tf.roll(image, shift=[-self.jitter_dx, -self.jitter_dy], axis=[1, 2])
        
        return image
    
    def random_roll(self, img, max_roll=128):
        """
        Apply a random shift to the image for tile processing
        
        Args:
            img: Input image tensor
            max_roll: Maximum roll amount
            
        Returns:
            tuple: (shift, rolled_img)
        """
        shift = [
            tf.random.uniform(shape=[], minval=-max_roll, maxval=max_roll+1, dtype=tf.int32),
            tf.random.uniform(shape=[], minval=-max_roll, maxval=max_roll+1, dtype=tf.int32)
        ]
        
        rolled_img = tf.roll(img, shift=shift, axis=[1, 2])
        
        return shift, rolled_img
    
    def calculate_loss(self, dream_features, ref_features, guidance_weight):
        """Loss calculation focused on channel correlations"""
        loss = 0

        layer_weights = np.linspace(0.5, 1.0, len(dream_features))
        
        for i, (dream_feat, ref_feat) in enumerate(zip(dream_features, ref_features)):
            standard_loss = tf.reduce_mean(tf.square(dream_feat))
            channels = tf.shape(dream_feat)[-1]
            ref_means = tf.reduce_mean(ref_feat, axis=[1, 2], keepdims=True)
            
            ref_std = tf.math.reduce_std(ref_means) + 1e-8
            ref_channels_norm = ref_means / ref_std

            dream_means = tf.reduce_mean(dream_feat, axis=[1, 2], keepdims=True)
            dream_std = tf.math.reduce_std(dream_means) + 1e-8
            dream_channels_norm = dream_means / dream_std
            
            channel_correlation = tf.reduce_mean(dream_channels_norm * ref_channels_norm)
            
            guidance_loss = channel_correlation
            
            layer_weight = layer_weights[i]
            combined_loss = layer_weight * ((1 - guidance_weight) * standard_loss + 
                            guidance_weight * guidance_loss)
            
            loss += combined_loss
            
        return loss
    
    def calculate_tiled_gradients(self, 
                              img, 
                              ref_features, 
                              guidance_weight=0.5, 
                              tile_size=512):
        img_shape = tf.shape(img)[1:3]
        img_height, img_width = int(img_shape[0]), int(img_shape[1])
        gradients = tf.Variable(tf.zeros_like(img))

        shift, img_rolled = self.random_roll(img, tile_size // 2)
        x_range = list(range(0, img_width, tile_size))
        y_range = list(range(0, img_height, tile_size))

        for y_start in y_range:
            for x_start in x_range:
                y_end = min(y_start + tile_size, img_height)
                x_end = min(x_start + tile_size, img_width)
                tile_h = y_end - y_start
                tile_w = x_end - x_start

                if tile_h < 75 or tile_w < 75:
                    print(f"Skipping small tile: {tile_h}x{tile_w}")
                    continue

                try:
                    img_tile = img_rolled[:, y_start:y_end, x_start:x_end, :]
                    with tf.GradientTape() as tape:
                        tape.watch(img_tile)
                        features = self.model(img_tile)
                        loss = self.calculate_loss(features, ref_features, guidance_weight)

                    tile_gradients = tape.gradient(loss, img_tile)
                    if tile_gradients is not None:
                        gradients[:, y_start:y_end, x_start:x_end, :].assign(tile_gradients)
                    else:
                        print(f"Gradient None for tile: {tile_h}x{tile_w}")

                except Exception as e:
                    print(f"Tile processing error at ({x_start},{y_start}): {e}")
                    continue

        gradients = tf.roll(gradients, shift=[-shift[0], -shift[1]], axis=[1, 2])
        gradients = gradients / (tf.math.reduce_std(gradients) + 1e-8)

        return gradients

    
    def process_with_tiling(self,
                           dream_img,
                           ref_imgs,
                           guidance_weight=0.5,
                           iterations=20,
                           step_size=0.01,
                           tile_size=512):
        """
        Process a large image using tiling approach
        
        Args:
            dream_img: Input image tensor as Variable
            ref_imgs: List of reference image tensors
            guidance_weight: How much to influence with reference (0-1)
            iterations: Number of optimization steps
            step_size: Size of each gradient step
            tile_size: Size of tiles to process
        
        Returns:
            Updated dream image
        """
        ref_features_list = []
        for img in ref_imgs:
            ref_features_list.append(self.model(img))

        ref_features = []
        for i in range(len(self.layers)):
            feature_stack = tf.stack([feat_list[i] for feat_list in ref_features_list], axis=0)
            ref_features.append(tf.reduce_mean(feature_stack, axis=0))

        lr_schedule = np.linspace(step_size, step_size * 0.6, iterations)

        dream_var = tf.Variable(dream_img)
        img_shape = tf.shape(dream_var)
        
        for i in range(iterations):
            current_step_size = lr_schedule[i]
            
            shape = tf.shape(dream_var)[1:3]
            if shape[0] < 2 or shape[1] < 2:
                print(f"Skipping tiling — image shape too small: {shape}")
                return dream_var

            gradients = self.calculate_tiled_gradients(
                dream_var, ref_features, guidance_weight, tile_size)

            dream_var.assign(tf.clip_by_value(
                dream_var + gradients * current_step_size, -1.0, 1.0))
            
            if i % 5 == 0:
                print(f"  Step {i+1}/{iterations} completed")
        
        return dream_var
    
    def process_single_octave(self, 
                             dream_img, 
                             ref_imgs,
                             guidance_weight=0.5,
                             iterations=20,
                             step_size=0.01,
                             jitter=32):
        """
        Process a single octave scale with improved gradient handling
        
        Args:
            dream_img: Input image tensor as Variable
            ref_imgs: List of reference image tensors
            guidance_weight: How much to influence with reference (0-1)
            iterations: Number of optimization steps
            step_size: Size of each gradient step
            jitter: Amount of random jitter to apply
        
        Returns:
            Updated dream image
        """
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

            if i % 5 == 0 and i > 0:
                with tf.GradientTape() as tv_tape:
                    tv_tape.watch(dream_var)
                    tv_loss = tf.image.total_variation(dream_var) / tf.cast(tf.reduce_prod(tf.shape(dream_var)[1:3]), tf.float32)
                
                tv_gradients = tv_tape.gradient(tv_loss, dream_var)
                tv_gradients = tv_gradients / (tf.math.reduce_std(tv_gradients) + 1e-8)

                dream_var.assign(tf.clip_by_value(
                    dream_var - tv_gradients * current_step_size * 0.1, -1.0, 1.0))
            
            if i % 5 == 0:
                try:
                    loss_value = float(loss.numpy())
                    print(f"  Step {i+1}/{iterations} - Loss: {loss_value:.4f}")
                except:
                    print(f"  Step {i+1}/{iterations} - Loss value not available")
        
        return dream_var
    
    def octave_process(self, 
                    base_img, 
                    ref_imgs,
                    octave_scale=1.3,
                    num_octaves=4,
                    guidance_weight=0.5,
                    iterations_per_octave=20,
                    step_size=0.01,
                    jitter=32,
                    show_progress=True):
        """
        Process the image at different scales for better detail.
        Following Google's DeepDream approach, processing from smallest to largest octave.
        
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
            scale_factor = tf.cast(octave_scale ** i, tf.float32)
            new_shape = tf.cast(shape_float / scale_factor, tf.int32)
            new_shape = tf.maximum(new_shape, 32)
            if new_shape[0] < 2 or new_shape[1] < 2:
                print(f"Skipping octave due to small shape: {new_shape}")
                continue
            octave_shapes.append(new_shape)
        
        octave_shapes = octave_shapes[::-1]
        
        if show_progress and num_octaves > 0:
            plt.figure(figsize=(15, 3 * num_octaves))

        shrunk_original_img = tf.image.resize(base_img, octave_shapes[0], 
                                             method=tf.image.ResizeMethod.LANCZOS3)

        dream_img = tf.identity(shrunk_original_img)
        
        for i, shape in enumerate(octave_shapes):
            print(f"Processing octave {i+1}/{num_octaves} - Size: {shape.numpy()}")
    
            if i > 0:
                dream_img = tf.image.resize(dream_img, shape, 
                                           method=tf.image.ResizeMethod.LANCZOS3)

                upscaled_shrunk_original = tf.image.resize(shrunk_original_img, shape,
                                                         method=tf.image.ResizeMethod.LANCZOS3)

                same_size_original = tf.image.resize(base_img, shape,
                                                   method=tf.image.ResizeMethod.LANCZOS3)
                
                lost_detail = same_size_original - upscaled_shrunk_original
                dream_img = dream_img + lost_detail

                shrunk_original_img = tf.identity(same_size_original)

            dream_var = tf.Variable(dream_img)

            resized_ref_imgs = [tf.image.resize(img, shape, 
                               method=tf.image.ResizeMethod.LANCZOS3) 
                               for img in ref_imgs]

            if tf.reduce_max(shape).numpy() > 512:
                print(f"  Using tiled processing for large image ({shape.numpy()})")
                result = self.process_with_tiling(
                    dream_var, 
                    resized_ref_imgs,
                    guidance_weight=guidance_weight,
                    iterations=iterations_per_octave,
                    step_size=step_size * (1.0 - 0.1 * i),
                    tile_size=512
                )
            else:
                result = self.process_single_octave(
                    dream_var, 
                    resized_ref_imgs,
                    guidance_weight=guidance_weight,
                    iterations=iterations_per_octave,
                    step_size=step_size * (1.0 - 0.1 * i),
                    jitter=min(jitter, tf.reduce_min(shape).numpy() // 8)
                )
            
            dream_img = tf.identity(result)
            
            if show_progress:
                try:
                    plt.subplot(num_octaves, 3, i*3 + 3)
                    plt.imshow(self.deprocess_image(result.numpy()))
                    plt.title(f"Octave {i+1} Result")
                    plt.axis("off")
                except Exception as e:
                    print(f"Warning: Could not display intermediate result: {e}")
        
        final_result = tf.image.resize(dream_img, original_shape, 
                                     method=tf.image.ResizeMethod.LANCZOS5)
        
        if show_progress and num_octaves > 0:
            try:
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display progress: {e}")
            
        return final_result
    
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
                         display_results=False,
                         show_progress=True,
                         high_quality=True):
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
        base_img = self.preprocess_image(base_img_path, process_size)
        
        ref_imgs = []
        for img_path in reference_img_paths:
            ref_imgs.append(self.preprocess_image(img_path, process_size))
        
        if display_results:
            num_refs = min(3, len(reference_img_paths))
            plt.figure(figsize=(5 * (num_refs + 1), 5))
            
            plt.subplot(1, num_refs + 1, 1)
            plt.imshow(original_img)
            plt.title("Base Image")
            plt.axis("off")
            
            for i in range(num_refs):
                plt.subplot(1, num_refs + 1, i + 2)
                plt.imshow(PIL.Image.open(reference_img_paths[i]))
                plt.title(f"Reference {i+1}")
                plt.axis("off")
                
            plt.tight_layout()
            plt.show()

        print(f"Starting guided dreaming with guidance weight: {guidance_weight}")
        print(f"Octaves: {num_octaves}, Scale: {octave_scale}, Iterations per octave: {iterations_per_octave}")
        
        result = self.octave_process(
            base_img, 
            ref_imgs,
            octave_scale=octave_scale,
            num_octaves=num_octaves,
            guidance_weight=guidance_weight,
            iterations_per_octave=iterations_per_octave,
            step_size=step_size,
            jitter=jitter,
            show_progress=show_progress
        )

        result_img = self.deprocess_image(result.numpy())

        if high_quality and result_img.shape[:2] != (orig_height, orig_width):
            print(f"Resizing final result to original dimensions: {original_size}")
            result_pil = PIL.Image.fromarray(result_img)
            result_pil = result_pil.resize(original_size, PIL.Image.LANCZOS)
            result_img = np.array(result_pil)
        
        if display_results:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title("Original Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(PIL.Image.open(reference_img_paths[0]))
            plt.title("Reference Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(result_img)
            plt.title(f"Guided Dream (weight: {guidance_weight})")
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()

        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            PIL.Image.fromarray(result_img).save(output_path)
            print(f"Result saved to {output_path}")
        
        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")
        
        return result_img