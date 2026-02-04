# Scope Shader Injector

A [Scope](https://github.com/daydreamlive/scope) plugin for real-time shader injection on video streams using Shadertoy-style GLSL.

## Features

- **Shadertoy-compatible syntax** - Write fragment shaders using the familiar `mainImage` function
- **Video input** - Process live video streams with custom shaders
- **Reference image support** - Blend or composite with a secondary image
- **Real-time parameters** - Adjust shader parameters while streaming
- **Text mode** - Preview shaders without video input

## Installation

### From Local Directory (Development)

1. Open Scope Desktop App
2. Go to Settings → Plugins
3. Browse to this plugin directory
4. Click Install

### From Git

```
https://github.com/yourusername/scope-shader-injector
```

## Usage

1. Select "Shader Injector" from the pipeline dropdown
2. Enter your GLSL shader code in the shader input field
3. Optionally load a reference image
4. Adjust parameters (time scale, mouse position, custom params)
5. Start streaming!

## Shader Format

Shaders use the Shadertoy format with a `mainImage` function:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Normalized pixel coordinates (0 to 1)
    vec2 uv = fragCoord / iResolution.xy;
    
    // Sample video texture
    vec4 video = texture(iChannel0, uv);
    
    // Sample reference image
    vec4 ref = texture(iChannel1, uv);
    
    // Mix based on time
    fragColor = mix(video, ref, 0.5 + 0.5 * sin(iTime));
}
```

## Available Uniforms

| Uniform | Type | Description |
|---------|------|-------------|
| `iResolution` | `vec3` | Viewport resolution (width, height, 1.0) |
| `iTime` | `float` | Time in seconds (affected by Time Scale parameter) |
| `iTimeDelta` | `float` | Time since last frame |
| `iFrame` | `int` | Current frame number |
| `iMouse` | `vec4` | Mouse position in pixels (x, y, click_x, click_y) |
| `iChannel0` | `sampler2D` | Video input texture |
| `iChannel1` | `sampler2D` | Reference image texture (or black if none) |
| `iChannel2` | `sampler2D` | Unused (black texture) |
| `iChannel3` | `sampler2D` | Unused (black texture) |
| `iParam1` | `float` | Custom parameter 1 (0-1 range) |
| `iParam2` | `float` | Custom parameter 2 (0-1 range) |
| `iParam3` | `float` | Custom parameter 3 (0-1 range) |

## Parameters

| Parameter | Description |
|-----------|-------------|
| **Shader Code** | Your GLSL fragment shader |
| **Reference Image** | Optional image for iChannel1 |
| **Time Scale** | Speed multiplier for iTime (0-10) |
| **Mouse X/Y** | Virtual mouse position (0-1) |
| **Param 1/2/3** | Custom float parameters for shader use |

## Example Shaders

The `examples/` directory contains ready-to-use shaders:

### Basic Effects
- `passthrough.glsl` - No modification (template)
- `invert.glsl` - Invert colors
- `grayscale.glsl` - Convert to grayscale
- `brightness_contrast.glsl` - Adjust brightness/contrast

### Color Effects
- `hue_shift.glsl` - Animated hue rotation
- `sepia.glsl` - Sepia tone
- `posterize.glsl` - Reduce color levels

### Distortion
- `wave_distortion.glsl` - Animated waves
- `pixelate.glsl` - Retro pixel look
- `barrel_distortion.glsl` - Lens distortion
- `kaleidoscope.glsl` - Mirror effect

### Visual Effects
- `vignette.glsl` - Dark corners
- `chromatic_aberration.glsl` - RGB split
- `edge_detection.glsl` - Sobel edges
- `scan_lines.glsl` - CRT effect

### Reference Image
- `blend_reference.glsl` - Blend modes
- `mask_with_reference.glsl` - Use as mask
- `difference.glsl` - Show differences

### Advanced
- `glitch.glsl` - Digital glitch
- `rain.glsl` - Rain overlay
- `matrix.glsl` - Matrix digital rain
- `fractal_overlay.glsl` - Mandelbrot blend

## Tips

1. **Use iParam1/2/3** for tweakable values - they update in real-time
2. **Multiply iTime** by different values for animation speed control
3. **Start simple** - complex shaders may reduce frame rate
4. **Test in text mode** first with the colorful test pattern
5. **Use iMouse** for interactive position-based effects

## Requirements

- Scope (with plugin support)
- Python 3.12+
- OpenGL 3.3+ capable GPU (or software rendering)

## Checking Dependencies

The plugin includes a built-in dependency checker. Run it to verify your system is properly configured:

```bash
# After installing the plugin, run:
python -m scope_shader_injector

# Or use the command-line tool:
shader-injector-check
```

This will show:
- Platform detection
- Available display (X11/Wayland)
- EGL/GL library availability
- Whether context creation succeeds

Example output:
```
============================================================
Scope Shader Injector - OpenGL Dependency Check
============================================================

Platform: linux
Display available: False
EGL library: libEGL.so.1
GL library: libGL.so.1
Context creation: EGL

✓ OpenGL context creation successful!
  The Shader Injector plugin should work correctly.

============================================================
```

## Headless Linux Setup

For running on a headless Linux server (no display), you need EGL support.

**Ubuntu 24.04+ (Noble):**
```bash
sudo apt-get update
sudo apt-get install -y libegl1 libegl-dev libgl1 libgl-dev
```

**Ubuntu 22.04 and earlier / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev
```

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install -y mesa-libEGL mesa-libEGL-devel mesa-libGL mesa-libGL-devel
```

**Arch Linux:**
```bash
sudo pacman -S mesa
```

**NVIDIA GPUs:**
```bash
# The NVIDIA driver should include EGL support
# Verify driver is working:
nvidia-smi

# If EGL is missing, it may be provided by:
sudo apt-get install -y libnvidia-gl-550  # Replace 550 with your driver version
```

**Docker with NVIDIA GPU:**
```bash
# Run container with GPU access and graphics capabilities
docker run --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  your-image

# Or use an NVIDIA base image with OpenGL support:
# FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04
```

**Docker without GPU (Software Rendering):**
```bash
# Add to Dockerfile:
RUN apt-get update && apt-get install -y \
    libegl1 libgl1 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for software rendering:
ENV LIBGL_ALWAYS_SOFTWARE=1
```

**Verify EGL Installation:**
```bash
# Check if EGL libraries are installed:
find /usr -name "libEGL*" 2>/dev/null

# Expected output:
# /usr/lib/x86_64-linux-gnu/libEGL.so.1
# /usr/lib/x86_64-linux-gnu/libEGL.so
```

The plugin automatically detects and uses the best available backend:
1. **EGL** (preferred for headless Linux with GPU)
2. **X11/GLX** (Linux with display)
3. **CGL** (macOS)
4. **WGL** (Windows)

## Dependencies

- moderngl - OpenGL shader compilation and rendering
- glcontext - EGL/GLX backend support for headless rendering  
- numpy - Array operations
- pillow - Image loading
- torch - Tensor operations (provided by Scope)

## Development

To modify the plugin during development:

1. Install as a local plugin
2. Make code changes
3. Click the Reload button in Settings → Plugins
4. The server will restart with your changes

### Running the Dependency Check During Development

```bash
cd /path/to/scope-shader-injector
python -m scope_shader_injector
```

## License

MIT License
