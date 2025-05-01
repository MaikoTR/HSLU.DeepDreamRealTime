import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import time
import os
import cv2
from tqdm import tqdm  # Use standard tqdm instead of notebook-specific
from IPython.display import HTML
from base64 import b64encode
import imageio
# Adjust the import based on your project structure
# If the GuidedDeepDream class is in another module:
try:
    from .guided_deepdream import GuidedDeepDream
except ImportError:
    try:
        from guided_deepdream import GuidedDeepDream
    except ImportError:
        # If importing directly fails, assume GuidedDeepDream is provided externally
        pass

class DeepDreamVideo:
    def __init__(self, 
                 guided_dreamer=None,
                 model_name='inception_v3',
                 layers=None):
        """
        Initialize the DeepDream Video generator.
        
        Args:
            guided_dreamer: An existing GuidedDeepDream instance or None to create new
            model_name: Name of the model to use (passed to GuidedDeepDream if created)
            layers: List of layer names for feature extraction (passed to GuidedDeepDream if created)
        """
        # Create a new GuidedDeepDream instance if not provided
        if guided_dreamer is None:
            self.dreamer = GuidedDeepDream(model_name=model_name, layers=layers)
        else:
            self.dreamer = guided_dreamer
            
        print("DeepDream Video generator initialized")
        
    def _extract_frames(self, video_path, target_fps=None, max_frames=None, resize_factor=1.0):
        """
        Extract frames from video file
        
        Args:
            video_path: Path to video file
            target_fps: Target framerate (None = use original)
            max_frames: Maximum number of frames to extract (None = no limit)
            resize_factor: Factor to resize frames (1.0 = original size)
            
        Returns:
            frames: List of PIL.Image objects
            fps: Actual frames per second
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
            
        # Get video properties
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate new dimensions
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        
        # Determine frame sampling
        fps = target_fps if target_fps is not None else orig_fps
        
        # Calculate frame sampling interval
        if target_fps is not None and target_fps < orig_fps:
            interval = orig_fps / target_fps
        else:
            interval = 1
            fps = orig_fps
            
        # Limit total frames if requested
        if max_frames is not None:
            frames_to_extract = min(max_frames, total_frames)
        else:
            frames_to_extract = total_frames
            
        print(f"Video properties: {width}x{height}, {orig_fps:.2f} fps, {total_frames} frames")
        print(f"Extracting {frames_to_extract} frames at {fps:.2f} fps")
        print(f"Resizing to {new_width}x{new_height}")
        
        # Extract frames
        frames = []
        count = 0
        frame_idx = 0
        
        with tqdm(total=frames_to_extract, desc="Extracting frames") as pbar:
            while cap.isOpened() and (max_frames is None or count < max_frames):
                success, frame = cap.read()
                if not success:
                    break
                    
                if frame_idx % interval < 1:  # Handle fractional intervals
                    # Convert BGR to RGB and resize
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if resize_factor != 1.0:
                        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), 
                                              interpolation=cv2.INTER_LANCZOS4)
                    
                    # Convert to PIL Image
                    pil_img = PIL.Image.fromarray(frame_rgb)
                    frames.append(pil_img)
                    count += 1
                    pbar.update(1)
                    
                frame_idx += 1
                if max_frames is not None and count >= max_frames:
                    break
        
        # Release the video capture
        cap.release()
        
        return frames, fps
    
    def _create_flow_field(self, frames, strength=1.0, smooth=0.8):
        """
        Create optical flow fields between consecutive frames
        
        Args:
            frames: List of PIL Images or numpy arrays
            strength: Flow strength factor
            smooth: Flow smoothing factor (0-1)
            
        Returns:
            flow_fields: List of flow fields between frames
        """
        flow_fields = []
        prev_gray = None
        
        # Convert PIL images to numpy if needed
        frame_arrays = []
        for frame in frames:
            if isinstance(frame, PIL.Image.Image):
                frame_arrays.append(np.array(frame))
            else:
                frame_arrays.append(frame)
        
        with tqdm(total=len(frame_arrays)-1, desc="Calculating optical flow") as pbar:
            for i in range(len(frame_arrays)):
                # Convert current frame to grayscale
                current = frame_arrays[i]
                current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
                
                if prev_gray is not None:
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, current_gray, None, 
                        pyr_scale=0.5, levels=3, winsize=15, 
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    
                    # Apply strength factor
                    flow = flow * strength
                    
                    # Store flow field
                    flow_fields.append(flow)
                    pbar.update(1)
                
                # Update previous frame with smoothing if enabled
                if smooth < 1.0 and prev_gray is not None:
                    prev_gray = cv2.addWeighted(current_gray, 1-smooth, prev_gray, smooth, 0)
                else:
                    prev_gray = current_gray.copy()
        
        return flow_fields
    
    def _warp_frame(self, frame, flow, blend_ratio=0.0):
        """
        Warp a frame using optical flow
        
        Args:
            frame: PIL Image or numpy array to warp
            flow: Optical flow field
            blend_ratio: Blend ratio between original and warped frame (0-1)
            
        Returns:
            warped_frame: Warped frame as PIL Image
        """
        # Convert PIL Image to numpy if needed
        if isinstance(frame, PIL.Image.Image):
            frame_array = np.array(frame)
        else:
            frame_array = frame
            
        # Create mapping grid
        h, w = flow.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Apply flow to grid
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        
        # Warp the image using the flow field
        warped = cv2.remap(frame_array, map_x, map_y, 
                           interpolation=cv2.INTER_LANCZOS4, 
                           borderMode=cv2.BORDER_REFLECT)
        
        # Blend original and warped if requested
        if blend_ratio > 0:
            warped = cv2.addWeighted(frame_array, blend_ratio, warped, 1-blend_ratio, 0)
            
        # Convert back to PIL Image
        return PIL.Image.fromarray(warped)
    
    def _save_temp_image(self, img, path):
        """Save a PIL Image or numpy array to a temporary file"""
        if isinstance(img, np.ndarray):
            PIL.Image.fromarray(img).save(path)
        else:
            img.save(path)
        return path
    
    def create_deepdream_video(self, 
                             video_path,
                             reference_img_paths,
                             output_path="deepdream_video.mp4",
                             temp_dir="temp_frames",
                             target_fps=None,
                             max_frames=None,
                             resize_factor=0.5,
                             guidance_weight=0.5,
                             octave_scale=1.3,
                             num_octaves=3,
                             iterations_per_octave=50,
                             step_size=0.01,
                             jitter=16,
                             flow_strength=1.0,
                             flow_blend=0.0,
                             temporal_smooth=0.5,
                             progressive_influence=False,
                             show_progress=True,
                             high_quality=True):
        """
        Create a DeepDream video by applying guided deepdream to video frames
        
        Args:
            video_path: Path to input video
            reference_img_paths: List of paths to reference images
            output_path: Path for output video
            temp_dir: Directory for temporary files
            target_fps: Target framerate (None = use original)
            max_frames: Maximum number of frames to process (None = all)
            resize_factor: Factor to resize frames (1.0 = original size)
            guidance_weight: How much to influence with reference (0-1)
            octave_scale: Scale factor between octaves
            num_octaves: Number of octave scales to process
            iterations_per_octave: Number of optimization steps per octave
            step_size: Size of each gradient step
            jitter: Amount of random jitter to apply
            flow_strength: Strength of optical flow warping (0 = disabled)
            flow_blend: Blend ratio between original and flow-warped frames
            temporal_smooth: Temporal smoothing factor between frames (0-1)
            progressive_influence: Whether to progressively increase influence
            show_progress: Whether to show processing progress
            high_quality: Whether to use higher quality settings
        """
        start_time = time.time()
        
        # Create temp directory if needed
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Extract frames from video
        print("Extracting video frames...")
        frames, fps = self._extract_frames(
            video_path, 
            target_fps=target_fps, 
            max_frames=max_frames,
            resize_factor=resize_factor
        )
        
        # Calculate optical flow if enabled
        flow_fields = []
        if flow_strength > 0:
            flow_fields = self._create_flow_field(
                frames, 
                strength=flow_strength,
                smooth=temporal_smooth
            )
        
        # Process reference images
        print("Processing reference images...")
        ref_imgs = []
        for img_path in reference_img_paths:
            ref_img = PIL.Image.open(img_path).convert("RGB")
            ref_imgs.append(ref_img)
        
        # Process all frames with DeepDream
        processed_frames = []
        prev_dream = None
        
        with tqdm(total=len(frames), desc="Creating DeepDream frames") as pbar:
            for i, frame in enumerate(frames):
                # Show occasional preview
                if show_progress and i % max(10, len(frames) // 5) == 0:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(frame)
                    plt.title(f"Original Frame {i}")
                    plt.axis("off")
                    
                    if len(processed_frames) > 0:
                        plt.subplot(1, 2, 2)
                        plt.imshow(processed_frames[-1])
                        plt.title(f"DeepDream Frame {len(processed_frames)-1}")
                        plt.axis("off")
                    
                    plt.tight_layout()
                    plt.show()
                
                # Apply flow-based warping if enabled
                if flow_strength > 0 and prev_dream is not None and i > 0:
                    # Warp previous dream frame with current flow
                    warped_dream = self._warp_frame(
                        prev_dream, 
                        flow_fields[i-1], 
                        blend_ratio=flow_blend
                    )
                    
                    # Save warped frame to temp file
                    warped_path = os.path.join(temp_dir, f"warped_{i:04d}.jpg")
                    self._save_temp_image(warped_dream, warped_path)
                    
                    # Use warped frame as base for next dream
                    base_path = warped_path
                else:
                    # Save current frame to temp file
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.jpg")
                    self._save_temp_image(frame, frame_path)
                    base_path = frame_path
                
                # Adjust guidance weight if progressive influence enabled
                if progressive_influence:
                    current_weight = guidance_weight * (i / len(frames))
                else:
                    current_weight = guidance_weight
                
                # Apply DeepDream to the frame
                dream_path = os.path.join(temp_dir, f"dream_{i:04d}.jpg")
                
                # Process with GuidedDeepDream
                dream_img = self.dreamer.guided_deepdream(
                    base_img_path=base_path,
                    reference_img_paths=reference_img_paths,
                    output_path=dream_path,
                    guidance_weight=current_weight,
                    octave_scale=octave_scale,
                    num_octaves=num_octaves,
                    iterations_per_octave=iterations_per_octave,
                    step_size=step_size,
                    jitter=jitter,
                    display_results=False,
                    show_progress=False,
                    high_quality=high_quality
                )
                
                # Load the dreamified image
                dream_pil = PIL.Image.fromarray(dream_img)
                processed_frames.append(dream_pil)
                
                # Store for next iteration
                prev_dream = dream_pil
                
                # Update progress bar
                pbar.update(1)
                
        # Generate video from processed frames
        print(f"Creating output video at {fps:.2f} fps...")
        
        # Convert all frames to numpy arrays
        frame_arrays = []
        for frame in processed_frames:
            if isinstance(frame, PIL.Image.Image):
                frame_arrays.append(np.array(frame))
            else:
                frame_arrays.append(frame)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame_arrays[0].shape[:2]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Write all frames
        for frame in frame_arrays:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
        # Release video writer
        video_writer.release()
        
        # Create a GIF for preview if requested
        gif_path = output_path.replace(".mp4", ".gif")
        imageio.mimsave(gif_path, frame_arrays, fps=min(fps, 10))
        
        # Report timing
        elapsed = time.time() - start_time
        print(f"Video creation completed in {elapsed:.2f} seconds")
        print(f"Output video saved to {output_path}")
        print(f"Preview GIF saved to {gif_path}")
        
        # Display preview in notebook if available
        try:
            from IPython.display import HTML
            return HTML(f'<video width="640" height="360" controls autoplay><source src="{output_path}" type="video/mp4"></video>')
        except:
            return None

    def create_deepdream_zoom_video(self,
                                   base_img_path,
                                   reference_img_paths,
                                   output_path="deepdream_zoom.mp4",
                                   temp_dir="temp_frames",
                                   frame_count=60,
                                   fps=30,
                                   zoom_speed=1.03,
                                   zoom_direction=None,
                                   rotation_per_frame=0.5,
                                   guidance_weight=0.5,
                                   octave_scale=1.3,
                                   num_octaves=3,
                                   iterations_per_octave=50,
                                   step_size=0.01,
                                   jitter=16,
                                   temporal_smooth=0.3,
                                   show_progress=True,
                                   high_quality=True):
        """
        Create a DeepDream zooming video effect from a single image
        
        Args:
            base_img_path: Path to base image
            reference_img_paths: List of paths to reference images
            output_path: Path for output video
            temp_dir: Directory for temporary files
            frame_count: Number of frames to generate
            fps: Frames per second for output video
            zoom_speed: Zoom factor per frame (>1 for zoom in, <1 for zoom out)
            zoom_direction: (x, y) tuple for zoom center (None = center)
            rotation_per_frame: Rotation angle per frame in degrees
            guidance_weight: How much to influence with reference (0-1)
            octave_scale: Scale factor between octaves
            num_octaves: Number of octave scales to process
            iterations_per_octave: Number of optimization steps per octave
            step_size: Size of each gradient step
            jitter: Amount of random jitter to apply
            temporal_smooth: Temporal smoothing factor between frames (0-1)
            show_progress: Whether to show processing progress
            high_quality: Whether to use higher quality settings
        """
        start_time = time.time()
        
        # Create temp directory if needed
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        # Load and analyze base image
        base_img = PIL.Image.open(base_img_path).convert("RGB")
        width, height = base_img.size
        
        # Determine zoom center if not specified
        if zoom_direction is None:
            center_x = width / 2
            center_y = height / 2
        else:
            center_x, center_y = zoom_direction
            
        print(f"Creating zoom video from {base_img_path}")
        print(f"Image dimensions: {width}x{height}, Zoom center: ({center_x}, {center_y})")
        print(f"Generating {frame_count} frames at {fps} fps")
        
        # Initialize variables for zoom sequence
        processed_frames = []
        prev_dream = None
        
        # Calculate total zoom for each frame
        zoom_factors = [zoom_speed ** i for i in range(frame_count)]
        
        # Generate zoomed frames
        with tqdm(total=frame_count, desc="Creating zoom frames") as pbar:
            for i in range(frame_count):
                # Calculate current zoom and rotation
                zoom = zoom_factors[i]
                angle = i * rotation_per_frame
                
                # Create larger canvas for zoomed image
                canvas_width = int(width * zoom)
                canvas_height = int(height * zoom)
                
                # Calculate offsets to center zoom point
                offset_x = int(center_x * zoom - center_x)
                offset_y = int(center_y * zoom - center_y)
                
                # Create transformation matrix
                M = cv2.getRotationMatrix2D((center_x, center_y), angle, zoom)
                M[0, 2] -= offset_x  # Adjust translation
                M[1, 2] -= offset_y
                
                # Convert base image to numpy array for transformation
                base_array = np.array(base_img)
                
                # Apply transformation
                warped = cv2.warpAffine(base_array, M, (width, height),
                                       flags=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_REPLICATE)
                
                # Convert back to PIL Image
                warped_img = PIL.Image.fromarray(warped)
                
                # Save warped frame
                warped_path = os.path.join(temp_dir, f"warped_{i:04d}.jpg")
                warped_img.save(warped_path)
                
                # Apply temporal smoothing if enabled
                if temporal_smooth > 0 and prev_dream is not None:
                    # Use previous dreamified frame as part of the base
                    smooth_path = os.path.join(temp_dir, f"smooth_{i:04d}.jpg")
                    
                    # Convert to numpy arrays for blending
                    prev_array = np.array(prev_dream)
                    warped_array = np.array(warped_img)
                    
                    # Blend previous dream with current warped frame
                    blended = cv2.addWeighted(
                        warped_array, 1-temporal_smooth,
                        prev_array, temporal_smooth,
                        0
                    )
                    
                    # Save blended frame
                    PIL.Image.fromarray(blended).save(smooth_path)
                    base_path = smooth_path
                else:
                    base_path = warped_path
                
                # Apply DeepDream to the frame
                dream_path = os.path.join(temp_dir, f"dream_{i:04d}.jpg")
                
                # Process with GuidedDeepDream
                dream_img = self.dreamer.guided_deepdream(
                    base_img_path=base_path,
                    reference_img_paths=reference_img_paths,
                    output_path=dream_path,
                    guidance_weight=guidance_weight,
                    octave_scale=octave_scale,
                    num_octaves=num_octaves,
                    iterations_per_octave=iterations_per_octave,
                    step_size=step_size,
                    jitter=jitter,
                    display_results=False,
                    show_progress=False,
                    high_quality=high_quality
                )
                
                # Load the dreamified image
                dream_pil = PIL.Image.fromarray(dream_img)
                processed_frames.append(dream_pil)
                
                # Store for next iteration
                prev_dream = dream_pil
                
                # Show occasional preview
                if show_progress and i % max(5, frame_count // 10) == 0:
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(warped_img)
                    plt.title(f"Zoomed Frame {i}")
                    plt.axis("off")
                    
                    plt.subplot(1, 2, 2)
                    plt.imshow(dream_pil)
                    plt.title(f"DeepDream Frame {i}")
                    plt.axis("off")
                    
                    plt.tight_layout()
                    plt.show()
                
                # Update progress bar
                pbar.update(1)
                
        # Generate video from processed frames
        print(f"Creating output video at {fps} fps...")
        
        # Convert all frames to numpy arrays
        frame_arrays = []
        for frame in processed_frames:
            if isinstance(frame, PIL.Image.Image):
                frame_arrays.append(np.array(frame))
            else:
                frame_arrays.append(frame)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = frame_arrays[0].shape[:2]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Write all frames
        for frame in frame_arrays:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
        # Release video writer
        video_writer.release()
        
        # Create a GIF for preview if requested
        gif_path = output_path.replace(".mp4", ".gif")
        imageio.mimsave(gif_path, frame_arrays, fps=min(fps, 10))
        
        # Report timing
        elapsed = time.time() - start_time
        print(f"Video creation completed in {elapsed:.2f} seconds")
        print(f"Output video saved to {output_path}")
        print(f"Preview GIF saved to {gif_path}")
        
        # Display preview in notebook if available
        try:
            from IPython.display import HTML
            return HTML(f'<video width="640" height="360" controls autoplay><source src="{output_path}" type="video/mp4"></video>')
        except:
            return None