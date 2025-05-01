#!/usr/bin/env python
# coding: utf-8

"""
Improved DeepDream Implementation in PyTorch

This implementation closely follows the original Caffe DeepDream implementation
to produce visually similar results, but uses PyTorch for easier installation
and compatibility with modern Python environments.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import scipy.ndimage as nd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepDreamModel:
    def __init__(self):
        # Load pre-trained GoogleNet (Inception v1)
        self.model = models.googlenet(pretrained=True)
        self.model.to(device)
        self.model.eval()
        
        # Store outputs of layers
        self.outputs = {}
        self.hooks = []
        
        # Register hooks for intermediate layers
        self._register_hooks()
        
        # These values closely match the original Caffe implementation
        # The original Caffe model used mean subtraction with these values
        self.mean = np.array([104.0, 116.0, 122.0]) / 255.0  # BGR mean in [0,1]
    
    def _register_hooks(self):
        """Register hooks to capture outputs of network layers."""
        def hook_function(name):
            def hook(module, input, output):
                self.outputs[name] = output
            return hook
        
        # Clear any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Get layer mapping similar to Caffe model
        layer_mapping = {
            # Inception3 modules
            'inception_3a/output': self.model.inception3a,
            'inception_3b/output': self.model.inception3b,
            
            # Inception4 modules
            'inception_4a/output': self.model.inception4a,
            'inception_4b/output': self.model.inception4b,
            'inception_4c/output': self.model.inception4c,
            'inception_4d/output': self.model.inception4d,
            'inception_4e/output': self.model.inception4e,
            
            # Inception5 modules
            'inception_5a/output': self.model.inception5a, 
            'inception_5b/output': self.model.inception5b,
        }
        
        # Register branch-specific hooks (similar to Caffe)
        branch_mapping = {
            # 3a branches
            'inception_3a/1x1': (self.model.inception3a, 'branch1'),
            'inception_3a/3x3_reduce': (self.model.inception3a, 'branch2.0'),
            'inception_3a/3x3': (self.model.inception3a, 'branch2.1'),
            'inception_3a/5x5_reduce': (self.model.inception3a, 'branch3.0'),
            'inception_3a/5x5': (self.model.inception3a, 'branch3.1'),
            'inception_3a/pool_proj': (self.model.inception3a, 'branch4.1'),
            
            # 3b branches
            'inception_3b/1x1': (self.model.inception3b, 'branch1'),
            'inception_3b/3x3_reduce': (self.model.inception3b, 'branch2.0'),
            'inception_3b/3x3': (self.model.inception3b, 'branch2.1'),
            'inception_3b/5x5_reduce': (self.model.inception3b, 'branch3.0'),
            'inception_3b/5x5': (self.model.inception3b, 'branch3.1'),
            'inception_3b/pool_proj': (self.model.inception3b, 'branch4.1'),
            
            # 4 modules (similar pattern)
            'inception_4a/5x5_reduce': (self.model.inception4a, 'branch3.0'),
            'inception_4b/5x5_reduce': (self.model.inception4b, 'branch3.0'),
            'inception_4c/5x5_reduce': (self.model.inception4c, 'branch3.0'),
            'inception_4d/5x5_reduce': (self.model.inception4d, 'branch3.0'),
            'inception_4e/5x5_reduce': (self.model.inception4e, 'branch3.0'),
        }
        
        # Register hooks for full inception modules
        for name, module in layer_mapping.items():
            handle = module.register_forward_hook(hook_function(name))
            self.hooks.append(handle)
        
        # Register hooks for specific branches
        for name, (parent, branch_path) in branch_mapping.items():
            # Convert string path to attribute access
            parts = branch_path.split('.')
            module = parent
            for part in parts:
                module = getattr(module, part)
            
            handle = module.register_forward_hook(hook_function(name))
            self.hooks.append(handle)
    
    def preprocess(self, img_array):
        """Convert a numpy array (HxWxC, RGB) to a PyTorch tensor (CxHxW, BGR)."""
        # Convert to float and normalize to [0, 1]
        img_float = img_array.astype(np.float32) / 255.0
        
        # Convert RGB to BGR (to match Caffe)
        img_bgr = img_float[:, :, ::-1]
        
        # Transpose to CxHxW
        img_t = img_bgr.transpose(2, 0, 1)
        
        # Convert to tensor
        tensor = torch.from_numpy(img_t).to(device)
        
        # Subtract mean (ImageNet mean used in Caffe model)
        for i in range(3):
            tensor[i] = tensor[i] - self.mean[i]
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def deprocess(self, tensor):
        """Convert a PyTorch tensor (1xCxHxW) back to a numpy array (HxWxC, RGB)."""
        # Move to CPU, remove batch dimension, convert to numpy
        img = tensor.squeeze().cpu().detach().numpy()
        
        # Add mean back
        for i in range(3):
            img[i] = img[i] + self.mean[i]
        
        # Transpose from CxHxW to HxWxC
        img = img.transpose(1, 2, 0)
        
        # Convert BGR to RGB
        img = img[:, :, ::-1]
        
        # Scale to 0-255 and clip
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img
    
    def objective_L2(self, features):
        """Simple L2 objective function to maximize feature activations."""
        return torch.nn.MSELoss()(features, torch.zeros_like(features))
    
    def make_step(self, img_tensor, end_layer, step_size=1.5, jitter=32, objective=None):
        """
        Perform one step of gradient ascent on the input image.
        This closely follows the original Caffe implementation's make_step function.
        """
        # Make a copy of the tensor that requires gradients
        input_tensor = img_tensor.clone().detach().requires_grad_(True)
        
        # Apply jitter by rolling the image (helps create cleaner results)
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        input_shifted = torch.roll(torch.roll(input_tensor, shifts=ox, dims=3), shifts=oy, dims=2)
        
        # Forward pass through the model
        self.model(input_shifted)
        
        # Get the target layer activations
        target_activations = self.outputs[end_layer]
        
        # Calculate loss
        if objective is None:
            # By default, maximize the L2 norm (similar to Caffe implementation)
            loss = -torch.norm(target_activations)
        else:
            loss = objective(target_activations)
        
        # Backward pass
        loss.backward()
        
        # Get the gradients
        grad = input_shifted.grad.clone()
        input_shifted.grad.zero_()
        
        # Normalize gradients (important for stable and visually pleasing results)
        grad_mean = torch.abs(grad).mean()
        if grad_mean > 0:
            grad = grad / (grad_mean + 1e-8)
        
        # Update the image using normalized gradients
        # Note: we use -= because our loss is negated (we maximize activations)
        updated = input_shifted.data - step_size * grad
        
        # Undo the jitter shift
        result = torch.roll(torch.roll(updated, shifts=-ox, dims=3), shifts=-oy, dims=2)
        
        return result
    
    def objective_guide(self, target_activations, guide_features):
        """
        Guide objective that maximizes similarity to guide features.
        This follows the approach from the original Caffe implementation.
        """
        # Get shapes
        batch, ch, h, w = target_activations.shape
        
        # Reshape activations to channel x space
        x = target_activations.view(ch, -1)
        y = guide_features.view(ch, -1)
        
        # Compute matrix of dot-products with guide features
        A = torch.mm(x.T, y)
        
        # Get indices of maximum similarity
        max_idx = torch.argmax(A, dim=1)
        
        # Reconstruct target with guide features
        matched = y[:, max_idx].view(1, ch, h, w)
        
        # Loss is negative so that we maximize similarity
        return -torch.sum(target_activations * matched)

def deepdream(model, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end_layer='inception_4c/output', step_size=1.5, jitter=32,
              guide_img=None, show_progress=True):
    """
    Apply the DeepDream algorithm to generate dream-like images.
    This is a PyTorch implementation of the original Caffe deepdream function.
    
    Args:
        model: The DeepDreamModel instance
        base_img: Starting image as numpy array (HxWxC, RGB, 0-255)
        iter_n: Number of iterations per octave
        octave_n: Number of octaves to process
        octave_scale: Scale between octaves
        end_layer: Target layer to optimize
        step_size: Gradient ascent step size
        jitter: Amount of jitter for shift augmentation
        guide_img: Optional guide image for guided dreaming
        show_progress: Whether to display progress during processing
    
    Returns:
        The resulting image as numpy array (HxWxC, RGB, 0-255)
    """
    # Extract guide features if provided
    guide_features = None
    if guide_img is not None:
        guide_tensor = model.preprocess(guide_img)
        with torch.no_grad():
            model.model(guide_tensor)
            guide_features = model.outputs[end_layer].clone()
    
    # Prepare base images for all octaves
    octaves = [model.preprocess(base_img).cpu().numpy()]
    
    # Create image pyramid by downscaling
    for i in range(octave_n - 1):
        octave = nd.zoom(octaves[-1], (1, 1, 1.0/octave_scale, 1.0/octave_scale), order=1)
        octaves.append(octave)
    
    # Array to store details generated in each octave
    detail = np.zeros_like(octaves[-1])
    
    # Process each octave from smallest to largest
    for octave, octave_base in enumerate(reversed(octaves)):
        # Move octave base back to device
        octave_base = torch.from_numpy(octave_base).to(device)
        
        h, w = octave_base.shape[2:4]
        print(f"Processing octave {octave+1}/{octave_n} at resolution {h}x{w}")
        
        if octave > 0:
            # Upscale detail from previous octave
            h1, w1 = detail.shape[2:4]
            detail = nd.zoom(detail, (1, 1, 1.0*h/h1, 1.0*w/w1), order=1)
            # Convert detail to tensor
            detail = torch.from_numpy(detail).to(device)
        else:
            detail = torch.zeros_like(octave_base)
        
        # Add detail to the octave base
        input_tensor = octave_base + detail
        
        # Perform gradient ascent iterations
        for i in tqdm(range(iter_n), disable=not show_progress):
            # Set up the objective function
            if guide_features is not None:
                objective = lambda x: model.objective_guide(x, guide_features)
            else:
                objective = None
            
            # Make a gradient ascent step
            input_tensor = model.make_step(
                input_tensor, end_layer, 
                step_size=step_size, 
                jitter=jitter,
                objective=objective
            )
            
            # Display progress if requested
            if show_progress and (i == 0 or i == iter_n - 1 or (i + 1) % 10 == 0):
                # Apply clipping to avoid extreme values
                vis = model.deprocess(input_tensor)
                plt.figure(figsize=(12, 8))
                plt.imshow(vis)
                plt.title(f"Octave {octave+1}, Iteration {i+1}")
                plt.axis('off')
                plt.show()
                plt.close()
        
        # Extract details from this octave
        detail = input_tensor - octave_base
        detail = detail.cpu().numpy()
    
    # Final result
    return model.deprocess(input_tensor)

def create_deepdream_animation(model, base_img, frame_count=100, zoom=0.05, 
                              layer='inception_4c/output', iter_n=10, octave_n=4, 
                              octave_scale=1.4, step_size=1.5, output_dir='frames'):
    """Create a DeepDream zoom animation."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize with base image
    frame = base_img.copy()
    h, w = frame.shape[:2]
    
    # Create frames
    for i in range(frame_count):
        print(f"Processing frame {i+1}/{frame_count}")
        
        # Apply deep dream
        frame = deepdream(
            model, frame, 
            iter_n=iter_n, 
            octave_n=octave_n, 
            octave_scale=octave_scale, 
            end_layer=layer, 
            step_size=step_size,
            show_progress=False
        )
        
        # Save the frame
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        Image.fromarray(frame).save(frame_path)
        
        # Zoom in
        frame = nd.affine_transform(
            frame, 
            [1-zoom, 1-zoom, 1], 
            [h*zoom/2, w*zoom/2, 0], 
            order=1
        )
    
    print(f"Animation frames saved to {output_dir}/")
    return frame

def display_image(img, title=None):
    """Display an image using matplotlib."""
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    """Command-line interface for DeepDream."""
    parser = argparse.ArgumentParser(description="DeepDream in PyTorch")
    
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--output', type=str, help='Path to save the output image')
    parser.add_argument('--layer', type=str, default='inception_4c/output', 
                        help='Layer to optimize (default: inception_4c/output)')
    parser.add_argument('--octaves', type=int, default=4, 
                        help='Number of octaves to process (default: 4)')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations per octave (default: 10)')
    parser.add_argument('--step-size', type=float, default=1.5,
                        help='Step size for gradient ascent (default: 1.5)')
    parser.add_argument('--guide', type=str, 
                        help='Path to guide image for guided dreaming')
    parser.add_argument('--animate', action='store_true',
                        help='Create a zooming animation')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames for animation (default: 100)')
    parser.add_argument('--zoom', type=float, default=0.05,
                        help='Scale factor for animation zoom (default: 0.05)')
    
    args = parser.parse_args()
    
    # Initialize model
    model = DeepDreamModel()
    
    # Load input image
    print(f"Loading input image: {args.image_path}")
    base_img = np.array(Image.open(args.image_path).convert('RGB'))
    display_image(base_img, "Input Image")
    
    # Load guide image if provided
    guide_img = None
    if args.guide:
        print(f"Loading guide image: {args.guide}")
        guide_img = np.array(Image.open(args.guide).convert('RGB'))
        display_image(guide_img, "Guide Image")
    
    if args.animate:
        print("Creating animation...")
        result = create_deepdream_animation(
            model, base_img, 
            frame_count=args.frames,
            zoom=args.zoom,
            layer=args.layer,
            iter_n=args.iterations,
            octave_n=args.octaves,
            step_size=args.step_size,
            output_dir='dream_frames'
        )
        print("Animation completed. Frames saved in 'dream_frames' directory.")
    else:
        print("Generating DeepDream image...")
        result = deepdream(
            model, base_img,
            iter_n=args.iterations,
            octave_n=args.octaves,
            octave_scale=1.4,
            end_layer=args.layer,
            step_size=args.step_size,
            guide_img=guide_img
        )
        
        # Display result
        display_image(result, "DeepDream Result")
        
        # Save result if output path is provided
        if args.output:
            Image.fromarray(result).save(args.output)
            print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()