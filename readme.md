# PyTorch DeepDream

This repository provides a modern PyTorch implementation of Google's DeepDream algorithm, originally described in their [blog post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html).

Unlike the original implementation which used Caffe, this version uses PyTorch for easier installation and better compatibility with modern Python environments.

## Features

- Complete PyTorch reimplementation of the DeepDream algorithm
- Support for guided dreaming (using a guide image)
- Multiple octave processing to create fine details
- Support for animation generation (zoom-in effect)
- Command-line interface
- Compatible with GPU acceleration

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/pytorch-deepdream.git
   cd pytorch-deepdream
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install torch torchvision tqdm numpy scipy pillow
   ```

   Note: For GPU support, make sure to install the appropriate CUDA version compatible with your GPU and PyTorch. See [PyTorch installation instructions](https://pytorch.org/get-started/locally/) for details.

## Usage

### Command-line Interface

The simplest way to use the script is through the command line:

```bash
python deepdream_simple.py input_image.jpg
```

This will process the image with default parameters and save the result as `input_image_dream.jpg`.

### Advanced Options

The script provides several options to customize the dreaming process:

```bash
python deepdream_simple.py input_image.jpg \
  --output output_image.jpg \
  --layer inception_4c/output \
  --octaves 6 \
  --octave_scale 1.4 \
  --iterations 10 \
  --lr 0.01 \
  --guide guide_image.jpg
```

### Creating Animations

To create a zoom animation similar to the original Deep Dream examples:

```bash
python deepdream_simple.py input_image.jpg --animate --frames 30 --zoom 0.05
```

This will create 30 frames of animation in the `dream_frames` directory, zooming in by a factor of 0.05 between frames.

### Available Layers

The GoogleNet model in PyTorch has layers that roughly correspond to the ones in the original Caffe model. Some of the available layers are:

- `inception_3a/output`
- `inception_3b/output`
- `inception_4a/output`
- `inception_4b/output`
- `inception_4c/output` (default)
- `inception_4d/output`
- `inception_4e/output`
- `inception_5a/output`
- `inception_5b/output`

You can also access branch-specific layers like:
- `inception_3a/3x3_reduce`
- `inception_4c/5x5_reduce`

Different layers produce different visual effects:
- Lower layers (3a, 3b) tend to enhance edges and textures, creating an impressionist feeling
- Middle layers (4a-4c) produce more complex patterns and shapes
- Higher layers (4d, 5a, 5b) tend to generate more recognizable object features

## Example Outputs

Different layers produce different types of visual hallucinations:

| Layer | Effect |
|-------|--------|
| inception_3b/output | Edges and textures, impressionist style |
| inception_4c/output | Complex patterns and shapes (default) |
| inception_5a/output | Higher-level features, more recognizable objects |

## Implementation Details

This PyTorch implementation replicates the key techniques from the original paper:

1. **Gradient Ascent** - The core technique is gradient ascent that maximizes the L2 norm of activations in a particular layer.

2. **Random Jitter** - Small random translations are applied before each optimization step to prevent grid-like artifacts.

3. **Octave Processing** - The image is processed at multiple scales (octaves) to generate both fine and coarse details.

4. **Guided Dreaming** - You can provide a guide image to influence the style of the hallucinations.

### Differences from Caffe Implementation

- Uses PyTorch's pre-trained GoogleNet model instead of Caffe's GoogLeNet
- Uses PyTorch's automatic differentiation instead of manual gradient calculations
- Simplified API and better documentation
- More Pythonic code organization
- More readable implementation of guide image functionality

## Using as a Module

You can also use the DeepDream class in your own code:

```python
from PIL import Image
import numpy as np
from deepdream_simple import deep_dream

# Basic usage
result_img = deep_dream('input.jpg', output_path='output.jpg')

# Guided dreaming with custom parameters
result_img = deep_dream(
    'input.jpg',
    output_path='guided_output.jpg',
    layer='inception_3b/output',
    octave_n=8,
    iterations=20,
    guide_path='guide.jpg'
)

# Create animation
result_img = deep_dream(
    'input.jpg',
    create_animation=True,
    animation_frames=40,
    animation_scale=0.03
)
```

## Tips for Getting Good Results

1. **Image Selection**: Start with images that have interesting textures or patterns.

2. **Layer Selection**: 
   - For texture enhancement: use early layers like `inception_3a/output`
   - For complex patterns: use middle layers like `inception_4c/output`
   - For object-like features: use deeper layers like `inception_5a/output`

3. **Octave Setting**: 
   - More octaves (6-8) produce finer details but take longer
   - Fewer octaves (3-4) are faster but produce coarser results

4. **Iteration Count**: 
   - More iterations (15-20) intensify the effect but may oversaturate
   - Fewer iterations (5-10) produce more subtle effects

5. **Guide Images**: For guided dreaming, choose guide images with strong patterns or features you want to emphasize.