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
        
        # Store random jitter values for consistent add/remove operations
        self.jitter_dx = 0
        self.jitter_dy = 0
        
        # Initialize the model
        if model_name == 'inception_v3':
            base_model = InceptionV3(include_top=False, weights='imagenet')
            
            # Default layers if none specified
            if layers is None:
                self.layers = ['mixed3', 'mixed5', 'mixed7']
            else:
                self.layers = layers
                
            # Create feature extraction model
            outputs = [base_model.get_layer(name).output for name in self.layers]
            self.model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            
            # The standard InceptionV3 input size is 299x299, but we'll keep this flexible
            # and use it only as a minimum constraint
            self.min_img_size = (299, 299)
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        print(f"Model initialized with layers: {self.layers}")
    
    def preprocess_image(self, img_path, target_size=None):
        """Load and preprocess image for the model"""
        # Open the image and convert to RGB to ensure 3 channels
        img = PIL.Image.open(img_path).convert("RGB")
        
        # Store original size
        original_size = img.size
        
        # If no target size provided, use the original size but ensure minimum dimensions
        if target_size is None:
            target_size = (max(original_size[0], self.min_img_size[0]),
                          max(original_size[1], self.min_img_size[1]))
        
        # Resize with high-quality LANCZOS resampling
        img = img.resize(target_size, PIL.Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Store original shape before preprocessing
        self.original_shape = img_array.shape[:-1]  # Store height, width (exclude channels)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply model-specific preprocessing
        if self.model_name == 'inception_v3':
            img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
            
        return img_array
    
    def deprocess_image(self, img):
        """Convert processed image back to displayable format with improved quality"""
        img = img.copy()
        
        # Remove batch dimension
        img = img[0]
        
        # Reverse preprocessing based on model
        if self.model_name == 'inception_v3':
            img /= 2.0
            img += 0.5
            
        # Scale to 0-255 range
        img *= 255.0
        
        # Clip values and convert to proper image type
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
            # Generate random shift values within jitter range
            self.jitter_dx = tf.random.uniform(shape=[], minval=-jitter, maxval=jitter+1, dtype=tf.int32)
            self.jitter_dy = tf.random.uniform(shape=[], minval=-jitter, maxval=jitter+1, dtype=tf.int32)
            
            # Apply shift using tf.roll (more efficient than manual padding)
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
            # Remove the shift by applying the opposite shift
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
        # Create a random shift
        shift = [
            tf.random.uniform(shape=[], minval=-max_roll, maxval=max_roll+1, dtype=tf.int32),
            tf.random.uniform(shape=[], minval=-max_roll, maxval=max_roll+1, dtype=tf.int32)
        ]
        
        # Roll the image
        rolled_img = tf.roll(img, shift=shift, axis=[1, 2])
        
        return shift, rolled_img
    
    def calculate_loss(self, dream_features, ref_features, guidance_weight):
        """Loss calculation focused on channel correlations"""
        loss = 0
        
        # Weight the layers differently (deeper layers get more weight)
        layer_weights = np.linspace(0.5, 1.0, len(dream_features))
        
        for i, (dream_feat, ref_feat) in enumerate(zip(dream_features, ref_features)):
            # Standard DeepDream part - maximize activation
            standard_loss = tf.reduce_mean(tf.square(dream_feat))
            
            # Reference guidance using channel correlations
            # This approach focuses on activating similar channels, regardless of spatial position
            
            # Get the channel dimension
            channels = tf.shape(dream_feat)[-1]
            
            # Calculate channel-wise statistics for reference
            # Mean activation per channel, across spatial dimensions
            ref_means = tf.reduce_mean(ref_feat, axis=[1, 2], keepdims=True)
            
            # Normalize reference channels by their activation strength
            ref_std = tf.math.reduce_std(ref_means) + 1e-8
            ref_channels_norm = ref_means / ref_std
            
            # Calculate the same for dream features
            dream_means = tf.reduce_mean(dream_feat, axis=[1, 2], keepdims=True)
            dream_std = tf.math.reduce_std(dream_means) + 1e-8
            dream_channels_norm = dream_means / dream_std
            
            # Compute channel correlation
            # We want to activate the same channels that are active in the reference
            channel_correlation = tf.reduce_mean(dream_channels_norm * ref_channels_norm)
            
            # Use channel correlation as the guidance loss
            guidance_loss = channel_correlation
            
            # Combine losses with layer weight
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
        """
        Calculate gradients using tiles to handle large images efficiently.
        This is based on Google's approach for handling large images.
        
        Args:
            img: Input image tensor
            ref_features: List of reference feature tensors
            guidance_weight: How much to influence with reference (0-1)
            tile_size: Size of the tiles to process
            
        Returns:
            Gradients for the entire image
        """
        # Get image shape
        img_shape = tf.shape(img)[1:3]
        
        # Initialize the image gradients to zero
        gradients = tf.zeros_like(img)
        
        # Add random jitter to prevent tile seams
        shift, img_rolled = self.random_roll(img, tile_size // 2)
        
        # Calculate tile ranges
        x_range = tf.range(0, img_shape[1], tile_size)
        y_range = tf.range(0, img_shape[0], tile_size)
        
        # If only one tile would be created, process the whole image
        if len(x_range) <= 1 and len(y_range) <= 1:
            with tf.GradientTape() as tape:
                tape.watch(img_rolled)
                features = self.model(img_rolled)
                loss = self.calculate_loss(features, ref_features, guidance_weight)
            gradients = tape.gradient(loss, img_rolled)
        else:
            # Skip the last tile edge unless there's only one tile
            if len(x_range) > 1:
                x_range = x_range[:-1]
            if len(y_range) > 1:
                y_range = y_range[:-1]
                
            # Add the endpoints to ensure we cover the whole image
            x_range = tf.concat([x_range, [img_shape[1] - tile_size]], axis=0)
            y_range = tf.concat([y_range, [img_shape[0] - tile_size]], axis=0)
            
            # Process each tile
            for y in y_range:
                for x in x_range:
                    # Get the tile
                    img_tile = img_rolled[:, y:y+tile_size, x:x+tile_size, :]
                    
                    # Calculate gradients for this tile
                    with tf.GradientTape() as tape:
                        tape.watch(img_tile)
                        features = self.model(img_tile)
                        loss = self.calculate_loss(features, ref_features, guidance_weight)
                    
                    # Get tile gradients
                    tile_gradients = tape.gradient(loss, img_tile)
                    
                    # Update the image gradients for this tile
                    gradients = tf.tensor_scatter_nd_update(
                        gradients,
                        tf.constant([[0, y, x, 0]]), 
                        tf.reshape(tile_gradients, [1, tile_size, tile_size, 3])
                    )
        
        # Undo the random shift applied to the image gradients
        gradients = tf.roll(gradients, shift=[-shift[0], -shift[1]], axis=[1, 2])
        
        # Normalize the gradients
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
        # Calculate reference features once
        ref_features_list = []
        for img in ref_imgs:
            ref_features_list.append(self.model(img))
            
        # Average the features from all reference images
        ref_features = []
        for i in range(len(self.layers)):
            feature_stack = tf.stack([feat_list[i] for feat_list in ref_features_list], axis=0)
            ref_features.append(tf.reduce_mean(feature_stack, axis=0))
        
        # Learning rate schedule - decrease step size gradually
        lr_schedule = np.linspace(step_size, step_size * 0.6, iterations)
        
        # Set the shape explicitly as a Variable for gradient tracking
        dream_var = tf.Variable(dream_img)
        img_shape = tf.shape(dream_var)
        
        for i in range(iterations):
            current_step_size = lr_schedule[i]
            
            # Calculate tiled gradients
            gradients = self.calculate_tiled_gradients(
                dream_var, ref_features, guidance_weight, tile_size)
            
            # Apply gradients
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
        # Calculate reference features once (they don't change during optimization)
        ref_features_list = []
        for img in ref_imgs:
            ref_features_list.append(self.model(img))
            
        # Average the features from all reference images
        ref_features = []
        for i in range(len(self.layers)):
            feature_stack = tf.stack([feat_list[i] for feat_list in ref_features_list], axis=0)
            ref_features.append(tf.reduce_mean(feature_stack, axis=0))
        
        # Learning rate schedule - decrease step size gradually
        lr_schedule = np.linspace(step_size, step_size * 0.6, iterations)
        
        # Set the shape explicitly as a Variable for gradient tracking
        dream_var = tf.Variable(dream_img)
        
        for i in range(iterations):
            current_step_size = lr_schedule[i]
            
            # Add random jitter
            jittered_img = self.add_jitter(dream_var, jitter)
            
            # Gradient calculation
            with tf.GradientTape() as tape:
                tape.watch(jittered_img)
                # Forward pass through the model
                dream_features = self.model(jittered_img)
                
                # Calculate loss
                loss = self.calculate_loss(dream_features, ref_features, guidance_weight)
            
            # Compute gradients
            gradients = tape.gradient(loss, jittered_img)
            
            # Normalize gradients for stable updates
            # This helps prevent extreme values from dominating the update
            gradients = gradients / (tf.math.reduce_std(gradients) + 1e-8)
            
            # Apply gradients to update the jittered image
            jittered_updated = jittered_img + gradients * current_step_size
            
            # Remove jitter to get the updated dream image
            updated_img = self.remove_jitter(jittered_updated, jitter)
            
            # Clip values to valid range and update the dream image
            dream_var.assign(tf.clip_by_value(updated_img, -1.0, 1.0))
            
            # Apply a small amount of total variation regularization to reduce high-frequency noise
            if i % 5 == 0 and i > 0:
                with tf.GradientTape() as tv_tape:
                    tv_tape.watch(dream_var)
                    # Calculate total variation loss
                    tv_loss = tf.image.total_variation(dream_var) / tf.cast(tf.reduce_prod(tf.shape(dream_var)[1:3]), tf.float32)
                
                # Get TV gradients
                tv_gradients = tv_tape.gradient(tv_loss, dream_var)
                tv_gradients = tv_gradients / (tf.math.reduce_std(tv_gradients) + 1e-8)
                
                # Apply a small amount of TV regularization
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
        # Get the original image shape
        original_shape = tf.shape(base_img)[1:3]
        shape_float = tf.cast(original_shape, tf.float32)
        
        # Keep the number of octaves reasonable
        num_octaves = min(num_octaves, 5)  # Limit to 5 octaves maximum
        
        # Generate octave pyramid shapes (from largest to smallest)
        octave_shapes = []
        min_dim = tf.reduce_min(shape_float).numpy()
        
        # Adjust num_octaves if the image is too small
        while octave_scale ** (num_octaves - 1) > min_dim / 100:
            num_octaves -= 1
            if num_octaves <= 1:
                num_octaves = 1
                break
                
        print(f"Using {num_octaves} octaves based on image size")
        
        # Calculate octave shapes from largest to smallest
        for i in range(num_octaves):
            scale_factor = tf.cast(octave_scale ** i, tf.float32)
            new_shape = tf.cast(shape_float / scale_factor, tf.int32)
            # Ensure minimum size of 100 pixels on each dimension
            new_shape = tf.maximum(new_shape, 100)
            octave_shapes.append(new_shape)
        
        # Reverse the order to process from smallest to largest (Google's approach)
        # Now octave_shapes[0] is the smallest and we'll work our way up
        octave_shapes = octave_shapes[::-1]
        
        if show_progress and num_octaves > 0:
            plt.figure(figsize=(15, 3 * num_octaves))
        
        # Create a shrunk version of the base image to match the smallest octave
        shrunk_original_img = tf.image.resize(base_img, octave_shapes[0], 
                                             method=tf.image.ResizeMethod.LANCZOS3)
        
        # Initialize the dream image with a copy of the shrunk original
        dream_img = tf.identity(shrunk_original_img)
        
        # Process each octave from smallest to largest (Google's approach)
        for i, shape in enumerate(octave_shapes):
            print(f"Processing octave {i+1}/{num_octaves} - Size: {shape.numpy()}")
            
            # For octaves after the first one, we need to resize and add lost detail
            if i > 0:
                # Resize the previous dream image to the current octave size
                dream_img = tf.image.resize(dream_img, shape, 
                                           method=tf.image.ResizeMethod.LANCZOS3)
                
                # Resize the shrunk original image to the current octave size
                upscaled_shrunk_original = tf.image.resize(shrunk_original_img, shape,
                                                         method=tf.image.ResizeMethod.LANCZOS3)
                
                # Resize the original base image to the current octave size
                same_size_original = tf.image.resize(base_img, shape,
                                                   method=tf.image.ResizeMethod.LANCZOS3)
                
                # Calculate the detail that was lost in the resizing process
                # This is the key to avoiding pixelation in Google's approach
                lost_detail = same_size_original - upscaled_shrunk_original
                
                # Add the lost detail back to the dream image
                dream_img = dream_img + lost_detail
                
                # Update shrunk_original_img for next iteration
                shrunk_original_img = tf.identity(same_size_original)
            
            # Make dream_img a Variable for gradient tracking
            dream_var = tf.Variable(dream_img)
            
            # Resize reference images to match current shape
            resized_ref_imgs = [tf.image.resize(img, shape, 
                               method=tf.image.ResizeMethod.LANCZOS3) 
                               for img in ref_imgs]
            
            # Check if we should use tiling (for large images)
            if tf.reduce_max(shape).numpy() > 512:
                print(f"  Using tiled processing for large image ({shape.numpy()})")
                result = self.process_with_tiling(
                    dream_var, 
                    resized_ref_imgs,
                    guidance_weight=guidance_weight,
                    iterations=iterations_per_octave,
                    step_size=step_size * (1.0 - 0.1 * i),  # Reduce step size for larger octaves
                    tile_size=512
                )
            else:
                # Process this octave normally
                result = self.process_single_octave(
                    dream_var, 
                    resized_ref_imgs,
                    guidance_weight=guidance_weight,
                    iterations=iterations_per_octave,
                    step_size=step_size * (1.0 - 0.1 * i),  # Reduce step size for larger octaves
                    jitter=min(jitter, tf.reduce_min(shape).numpy() // 8)  # Scale jitter with image size
                )
            
            # Update dream_img for next octave
            dream_img = tf.identity(result)
            
            # Show intermediate result for this octave
            if show_progress:
                try:
                    plt.subplot(num_octaves, 3, i*3 + 3)
                    plt.imshow(self.deprocess_image(result.numpy()))
                    plt.title(f"Octave {i+1} Result")
                    plt.axis("off")
                except Exception as e:
                    print(f"Warning: Could not display intermediate result: {e}")
        
        # For the final result, resize to original shape with high quality
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
        
        # Get original image for display and to determine dimensions
        original_img = PIL.Image.open(base_img_path).convert("RGB")
        orig_width, orig_height = original_img.size
        original_size = (orig_width, orig_height)
        
        # Calculate target size for processing if not provided
        if target_size is None:
            # Determine if we need to resize based on constraints
            if high_quality:
                # For high quality, use original size but ensure minimum dimensions
                process_size = (max(orig_width, self.min_img_size[0]), 
                               max(orig_height, self.min_img_size[1]))
            else:
                # For standard quality, use min dimensions
                process_size = self.min_img_size
        else:
            process_size = target_size
        
        print(f"Processing at size: {process_size}")
        
        # Load and preprocess images
        base_img = self.preprocess_image(base_img_path, process_size)
        
        ref_imgs = []
        for img_path in reference_img_paths:
            ref_imgs.append(self.preprocess_image(img_path, process_size))
        
        # Display original images
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
        
        # Process the image through octaves
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
        
        # Convert result to image
        result_img = self.deprocess_image(result.numpy())
        
        # When high quality is enabled and there's a size mismatch,
        # resize the result back to original dimensions with high quality
        if high_quality and result_img.shape[:2] != (orig_height, orig_width):
            print(f"Resizing final result to original dimensions: {original_size}")
            result_pil = PIL.Image.fromarray(result_img)
            result_pil = result_pil.resize(original_size, PIL.Image.LANCZOS)
            result_img = np.array(result_pil)
        
        if display_results:
            # Display final result
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
            
        # Save result if output path specified
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            PIL.Image.fromarray(result_img).save(output_path)
            print(f"Result saved to {output_path}")
        
        elapsed = time.time() - start_time
        print(f"Processing completed in {elapsed:.2f} seconds")
        
        return result_img