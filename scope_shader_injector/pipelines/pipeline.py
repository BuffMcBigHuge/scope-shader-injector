"""Shader Injector pipeline implementation."""

import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import moderngl
import numpy as np
import torch
from PIL import Image

from scope.core.pipelines.interface import Pipeline, Requirements

from .schema import DEFAULT_SHADER, ShaderInjectorConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


def _find_library_path(name: str) -> str | None:
    """Find the full path to a library.
    
    Args:
        name: Library name (e.g., 'EGL', 'GL')
        
    Returns:
        Full library path or None if not found.
    """
    import ctypes.util
    
    # Try standard find_library first
    lib = ctypes.util.find_library(name)
    if lib:
        return lib
    
    # Try common paths manually
    common_paths = [
        f'/usr/lib/x86_64-linux-gnu/lib{name}.so.1',
        f'/usr/lib/x86_64-linux-gnu/lib{name}.so',
        f'/usr/lib64/lib{name}.so.1',
        f'/usr/lib64/lib{name}.so',
        f'/usr/lib/lib{name}.so.1',
        f'/usr/lib/lib{name}.so',
        # NVIDIA paths
        f'/usr/lib/x86_64-linux-gnu/nvidia/current/lib{name}.so.1',
        f'/usr/lib64/nvidia/lib{name}.so.1',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def _check_egl_available() -> tuple[bool, str | None]:
    """Check if EGL libraries are available on the system.
    
    Returns:
        Tuple of (available, library_path)
    """
    if not sys.platform.startswith('linux'):
        return False, None
    
    try:
        lib_path = _find_library_path('EGL')
        return lib_path is not None, lib_path
    except Exception:
        return False, None


def _check_gl_available() -> tuple[bool, str | None]:
    """Check if GL libraries are available on the system.
    
    Returns:
        Tuple of (available, library_path)
    """
    if not sys.platform.startswith('linux'):
        return False, None
    
    try:
        lib_path = _find_library_path('GL')
        return lib_path is not None, lib_path
    except Exception:
        return False, None


def _check_display_available() -> bool:
    """Check if a display is available (X11 or Wayland)."""
    return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))


def _is_docker() -> bool:
    """Check if running inside a Docker container."""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'


def _get_platform_install_instructions() -> str:
    """Get platform-specific installation instructions for OpenGL support."""
    if sys.platform.startswith('linux'):
        in_docker = _is_docker()
        
        base_instructions = """
==============================================================================
OPENGL/EGL LIBRARIES NOT FOUND
==============================================================================

The Shader Injector plugin requires OpenGL/EGL libraries for GPU-accelerated
shader rendering. These are NOT installed on this system.

"""
        if in_docker:
            return base_instructions + """
DOCKER INSTALLATION:
-------------------
You need to rebuild your Docker image with EGL support. Add to your Dockerfile:

  # For Ubuntu/Debian based images:
  RUN apt-get update && apt-get install -y \\
      libegl1-mesa \\
      libgl1-mesa-glx \\
      libgles2-mesa \\
      && rm -rf /var/lib/apt/lists/*

  # For NVIDIA GPU support, use nvidia/cuda base image and add:
  ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility

Then run the container with GPU access:
  docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility ...

Alternative: Use a pre-built image with EGL:
  FROM nvidia/cuda:12.1-runtime-ubuntu22.04
  # or
  FROM nvidia/opengl:1.2-glvnd-runtime-ubuntu22.04
"""
        else:
            return base_instructions + """
INSTALLATION COMMANDS:
---------------------

Ubuntu/Debian:
  sudo apt-get update
  sudo apt-get install -y libegl1-mesa libegl1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev

Fedora/RHEL/CentOS:
  sudo dnf install -y mesa-libEGL mesa-libEGL-devel mesa-libGL mesa-libGL-devel

Arch Linux:
  sudo pacman -S mesa

With NVIDIA GPU (ensure driver is installed):
  # The nvidia-driver package should include EGL support
  # Verify with: ls /usr/lib/x86_64-linux-gnu/libEGL*

For software rendering (no GPU required):
  sudo apt-get install -y libosmesa6 mesa-utils
  export LIBGL_ALWAYS_SOFTWARE=1

After installing, restart the Scope server.
"""
    elif sys.platform == 'darwin':
        return """
macOS should work out of the box with the default CGL backend.
If you're running in a headless environment (e.g., SSH), you may need
to use a virtual display or run on a machine with a GPU.
"""
    elif sys.platform == 'win32':
        return """
Windows should work out of the box with the default WGL backend.
Ensure you have OpenGL drivers installed for your GPU.
"""
    return "Please ensure OpenGL drivers are installed for your platform."


def create_headless_context() -> moderngl.Context:
    """Create a ModernGL context that works in headless environments.

    Tries multiple backends in order of preference based on platform
    and available resources.

    Returns:
        ModernGL context.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    errors = []
    
    logger.info(f"Creating OpenGL context on platform: {sys.platform}")
    logger.info(f"Display available: {_check_display_available()}")
    
    if sys.platform.startswith('linux'):
        egl_available, egl_path = _check_egl_available()
        gl_available, gl_path = _check_gl_available()
        logger.info(f"EGL library: {egl_path or 'NOT FOUND'}")
        logger.info(f"GL library: {gl_path or 'NOT FOUND'}")

    # Strategy varies by platform
    if sys.platform.startswith('linux'):
        has_display = _check_display_available()
        egl_available, egl_path = _check_egl_available()
        gl_available, gl_path = _check_gl_available()
        
        # Try EGL with detected library paths first
        if egl_available and egl_path:
            try:
                logger.info(f"Attempting EGL backend with detected path: {egl_path}")
                kwargs = {'standalone': True, 'backend': 'egl'}
                if egl_path:
                    kwargs['libegl'] = egl_path
                if gl_path:
                    kwargs['libgl'] = gl_path
                ctx = moderngl.create_context(**kwargs)
                logger.info("Successfully created ModernGL context using EGL backend")
                return ctx
            except Exception as e:
                errors.append(f"EGL (detected paths): {e}")
                logger.warning(f"EGL backend with detected paths failed: {e}")
        
        # Try EGL with default library names
        try:
            logger.info("Attempting EGL backend with default libraries...")
            ctx = moderngl.create_context(standalone=True, backend='egl')
            logger.info("Successfully created ModernGL context using EGL backend (default)")
            return ctx
        except Exception as e:
            errors.append(f"EGL (default): {e}")
            logger.warning(f"EGL backend with default libraries failed: {e}")
        
        # Try EGL with common library paths
        egl_lib_variants = [
            'libEGL.so.1',
            'libEGL.so',
            '/usr/lib/x86_64-linux-gnu/libEGL.so.1',
            '/usr/lib64/libEGL.so.1',
        ]
        gl_lib_variants = [
            'libGL.so.1',
            'libGL.so',
            '/usr/lib/x86_64-linux-gnu/libGL.so.1',
            '/usr/lib64/libGL.so.1',
        ]
        
        for egl_lib in egl_lib_variants:
            for gl_lib in gl_lib_variants:
                try:
                    logger.info(f"Attempting EGL with libegl={egl_lib}, libgl={gl_lib}")
                    ctx = moderngl.create_context(
                        standalone=True,
                        backend='egl',
                        libegl=egl_lib,
                        libgl=gl_lib,
                    )
                    logger.info(f"Successfully created ModernGL context using EGL ({egl_lib})")
                    return ctx
                except Exception as e:
                    # Only log first few attempts to avoid spam
                    if len(errors) < 3:
                        errors.append(f"EGL ({egl_lib}): {e}")
                    logger.debug(f"EGL with {egl_lib}, {gl_lib} failed: {e}")
        
        # Try X11/GLX if display is available
        if has_display:
            try:
                logger.info("Attempting X11/GLX backend...")
                ctx = moderngl.create_context(standalone=True)
                logger.info("Successfully created ModernGL context using X11/GLX backend")
                return ctx
            except Exception as e:
                errors.append(f"X11/GLX: {e}")
                logger.warning(f"X11/GLX backend failed: {e}")
        else:
            errors.append("X11/GLX: No display available (DISPLAY not set)")

    elif sys.platform == 'darwin':
        # macOS uses CGL
        try:
            logger.info("Attempting CGL backend (macOS)...")
            ctx = moderngl.create_context(standalone=True)
            logger.info("Successfully created ModernGL context using CGL backend")
            return ctx
        except Exception as e:
            errors.append(f"CGL: {e}")
            logger.warning(f"CGL backend failed: {e}")

    elif sys.platform == 'win32':
        # Windows uses WGL
        try:
            logger.info("Attempting WGL backend (Windows)...")
            ctx = moderngl.create_context(standalone=True)
            logger.info("Successfully created ModernGL context using WGL backend")
            return ctx
        except Exception as e:
            errors.append(f"WGL: {e}")
            logger.warning(f"WGL backend failed: {e}")

    else:
        # Unknown platform, try default
        try:
            logger.info("Attempting default backend...")
            ctx = moderngl.create_context(standalone=True)
            logger.info("Successfully created ModernGL context using default backend")
            return ctx
        except Exception as e:
            errors.append(f"Default: {e}")
            logger.warning(f"Default backend failed: {e}")

    # All backends failed - provide helpful error message
    error_details = "; ".join(errors)
    install_instructions = _get_platform_install_instructions()
    
    raise RuntimeError(
        f"Failed to create OpenGL context.\n\n"
        f"Attempted backends: {error_details}\n\n"
        f"Installation instructions:{install_instructions}"
    )


# Vertex shader for fullscreen quad
VERTEX_SHADER = """
#version 330 core

in vec2 in_position;
in vec2 in_texcoord;
out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""

# Fragment shader template that wraps user's Shadertoy-style code
FRAGMENT_SHADER_TEMPLATE = """
#version 330 core

// Shadertoy-compatible uniforms
uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform vec4 iMouse;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

// Custom parameters
uniform float iParam1;
uniform float iParam2;
uniform float iParam3;

in vec2 v_texcoord;
out vec4 fragColor;

// User's shader code
{user_code}

void main() {{
    vec2 fragCoord = v_texcoord * iResolution.xy;
    mainImage(fragColor, fragCoord);
}}
"""


class ShaderInjectorPipeline(Pipeline):
    """Real-time shader injection pipeline using Shadertoy-style GLSL."""

    def __init__(
        self,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ):
        """Initialize the shader injector pipeline.

        Args:
            height: Output frame height in pixels.
            width: Output frame width in pixels.
            **kwargs: Additional arguments.
        """
        self.height = height
        self.width = width
        
        logger.info(f"Initializing ShaderInjectorPipeline: {width}x{height}")

        # Create standalone OpenGL context (headless-compatible)
        self.ctx = create_headless_context()
        
        # Log context info
        logger.info(f"OpenGL Version: {self.ctx.version_code}")
        logger.info(f"OpenGL Vendor: {self.ctx.info.get('GL_VENDOR', 'Unknown')}")
        logger.info(f"OpenGL Renderer: {self.ctx.info.get('GL_RENDERER', 'Unknown')}")

        # Create fullscreen quad geometry
        vertices = np.array([
            # position    texcoord
            -1.0, -1.0,   0.0, 0.0,
            1.0, -1.0,    1.0, 0.0,
            -1.0, 1.0,    0.0, 1.0,
            1.0, 1.0,     1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)

        # Initialize state
        self.program = None
        self.vao = None
        self.current_shader_code = None
        self.fbo = None
        self.output_texture = None

        # Textures for input channels
        self.channel_textures = [None, None, None, None]

        # Reference image texture
        self.reference_image_path = None
        self.reference_texture = None

        # Frame counter and timing
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        # Compile default shader
        self._compile_shader(DEFAULT_SHADER)
        self._setup_framebuffer()

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the configuration class for this pipeline."""
        return ShaderInjectorConfig

    def _compile_shader(self, shader_code: str) -> bool:
        """Compile a Shadertoy-style shader.

        Args:
            shader_code: User's GLSL shader code with mainImage function.

        Returns:
            True if compilation succeeded, False otherwise.
        """
        if shader_code == self.current_shader_code and self.program is not None:
            return True

        # Build full fragment shader
        fragment_shader = FRAGMENT_SHADER_TEMPLATE.format(user_code=shader_code)

        try:
            # Clean up old program
            if self.program is not None:
                self.program.release()
                self.program = None
            if self.vao is not None:
                self.vao.release()
                self.vao = None

            # Compile new program
            self.program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=fragment_shader,
            )

            # Create VAO
            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')],
            )

            self.current_shader_code = shader_code
            return True

        except Exception as e:
            print(f"Shader compilation error: {e}")
            # Revert to default shader if compilation fails
            if shader_code != DEFAULT_SHADER:
                return self._compile_shader(DEFAULT_SHADER)
            return False

    def _setup_framebuffer(self):
        """Set up the output framebuffer."""
        if self.output_texture is not None:
            self.output_texture.release()
        if self.fbo is not None:
            self.fbo.release()

        logger.info(f"Setting up framebuffer: {self.width}x{self.height}")
        
        try:
            self.output_texture = self.ctx.texture((self.width, self.height), 4)
            self.fbo = self.ctx.framebuffer(color_attachments=[self.output_texture])
            logger.info(f"Framebuffer created successfully")
        except Exception as e:
            logger.error(f"Failed to create framebuffer: {e}")
            raise
        
        # Test that we can use the framebuffer
        try:
            self.fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            logger.info("Framebuffer test passed")
        except Exception as e:
            logger.error(f"Framebuffer test failed: {e}")
            raise

    def _create_texture_from_tensor(self, tensor: torch.Tensor) -> moderngl.Texture:
        """Create a ModernGL texture from a PyTorch tensor.

        Args:
            tensor: Input tensor of shape (1, H, W, C) or (H, W, C).
                    Values can be in [0, 255] or [0, 1] range.

        Returns:
            ModernGL texture.
        """
        # Handle different tensor shapes
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Log tensor info for debugging
        logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, "
                     f"min: {tensor.min().item():.3f}, max: {tensor.max().item():.3f}")

        # Convert to numpy
        data = tensor.cpu().numpy()
        
        # Handle different shapes
        if data.ndim == 2:
            # Grayscale - expand to RGB
            h, w = data.shape
            c = 1
            data = np.stack([data, data, data], axis=-1)
        else:
            h, w, c = data.shape
        
        # Normalize to [0, 255] range if needed
        if data.max() <= 1.0 and data.min() >= 0.0:
            # Data is in [0, 1] range
            data = (data * 255.0)
        elif data.max() <= 1.0 and data.min() >= -1.0:
            # Data might be in [-1, 1] range
            data = ((data + 1.0) * 127.5)
        
        # Convert to uint8
        data = np.clip(data, 0, 255).astype(np.uint8)

        # Ensure RGBA format (4 channels)
        if c == 1 or data.shape[-1] == 1:
            # Grayscale to RGBA
            if data.ndim == 2:
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[:, :, 0] = data
                rgba[:, :, 1] = data
                rgba[:, :, 2] = data
                rgba[:, :, 3] = 255
            else:
                gray = data[:, :, 0]
                rgba = np.zeros((h, w, 4), dtype=np.uint8)
                rgba[:, :, 0] = gray
                rgba[:, :, 1] = gray
                rgba[:, :, 2] = gray
                rgba[:, :, 3] = 255
            data = rgba
        elif data.shape[-1] == 3:
            # RGB to RGBA
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = data
            rgba[:, :, 3] = 255
            data = rgba
        elif data.shape[-1] != 4:
            raise ValueError(f"Unsupported number of channels: {data.shape[-1]}")

        # Ensure contiguous C-order array
        data = np.ascontiguousarray(data, dtype=np.uint8)
        
        # Validate dimensions
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid texture dimensions: {w}x{h}")
        
        # Validate data size
        expected_size = w * h * 4
        actual_size = data.nbytes
        if actual_size != expected_size:
            raise ValueError(f"Data size mismatch: expected {expected_size}, got {actual_size}")
        
        logger.debug(f"Creating texture: {w}x{h}, data shape: {data.shape}, data size: {actual_size}")

        # Create texture
        try:
            # Flip vertically for OpenGL (origin at bottom-left)
            data = np.flipud(data).copy()
            
            texture = self.ctx.texture((w, h), 4, data.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
            return texture
        except Exception as e:
            logger.error(f"Failed to create texture: {e}")
            logger.error(f"Texture params: size=({w}, {h}), components=4, data_len={len(data.tobytes())}")
            logger.error(f"Context info: version={self.ctx.version_code}")
            raise

    def _load_reference_image(self, path: str) -> moderngl.Texture | None:
        """Load a reference image as a texture.

        Args:
            path: Path to the image file.

        Returns:
            ModernGL texture or None if loading fails.
        """
        if not path or not Path(path).exists():
            return None

        try:
            img = Image.open(path).convert('RGBA')
            img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            data = np.array(img, dtype=np.uint8)

            texture = self.ctx.texture((self.width, self.height), 4, data.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return texture

        except Exception as e:
            print(f"Failed to load reference image: {e}")
            return None

    def _create_black_texture(self) -> moderngl.Texture:
        """Create a black texture for unused channels."""
        data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        data[:, :, 3] = 255  # Full alpha

        texture = self.ctx.texture((self.width, self.height), 4, data.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def _create_test_pattern(self) -> moderngl.Texture:
        """Create a test pattern texture for text mode (no video input)."""
        # Create a colorful gradient pattern
        y, x = np.mgrid[0:self.height, 0:self.width]
        r = ((x / self.width) * 255).astype(np.uint8)
        g = ((y / self.height) * 255).astype(np.uint8)
        b = (((x + y) / (self.width + self.height)) * 255).astype(np.uint8)
        a = np.full((self.height, self.width), 255, dtype=np.uint8)

        data = np.stack([r, g, b, a], axis=-1)
        texture = self.ctx.texture((self.width, self.height), 4, data.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return texture

    def prepare(self, **kwargs) -> Requirements | None:
        """Declare input requirements.

        Returns:
            Requirements for video mode, None for text mode.
        """
        # Check if we're in video mode
        video_mode = kwargs.get("video", False)

        if video_mode:
            return Requirements(input_size=1)
        else:
            # Text mode - no video input needed
            return None

    def __call__(self, **kwargs) -> dict:
        """Process video frames with the shader.

        Args:
            **kwargs: Runtime parameters including:
                - video: List of input frame tensors (video mode)
                - shader_code: GLSL shader code
                - reference_image: Path to reference image
                - time_scale: Multiplier for iTime
                - mouse_x, mouse_y: Mouse position
                - param1, param2, param3: Custom parameters
                - prompt: Alternative source for shader code

        Returns:
            Dict with "video" key containing processed frames tensor.
        """
        # Get shader code from kwargs or prompt
        shader_code = kwargs.get("shader_code", kwargs.get("prompt", DEFAULT_SHADER))
        if not shader_code or shader_code.strip() == "":
            shader_code = DEFAULT_SHADER

        # Compile shader if changed
        self._compile_shader(shader_code)

        # Get parameters
        time_scale = kwargs.get("time_scale", 1.0)
        mouse_x = kwargs.get("mouse_x", 0.5)
        mouse_y = kwargs.get("mouse_y", 0.5)
        param1 = kwargs.get("param1", 0.5)
        param2 = kwargs.get("param2", 0.5)
        param3 = kwargs.get("param3", 0.5)

        # Handle reference image
        reference_image = kwargs.get("reference_image", "")
        if reference_image != self.reference_image_path:
            if self.reference_texture is not None:
                self.reference_texture.release()
            self.reference_texture = self._load_reference_image(reference_image)
            self.reference_image_path = reference_image

        # Get video input or create test pattern
        video = kwargs.get("video")
        if video is not None and len(video) > 0:
            # Video mode - use input frame
            frame = video[0]
            logger.debug(f"Processing video frame: type={type(frame)}, shape={frame.shape if hasattr(frame, 'shape') else 'N/A'}")
            try:
                input_texture = self._create_texture_from_tensor(frame)
            except Exception as e:
                logger.error(f"Failed to create texture from video frame: {e}")
                logger.error(f"Frame info: type={type(frame)}, shape={frame.shape if hasattr(frame, 'shape') else 'N/A'}, "
                            f"dtype={frame.dtype if hasattr(frame, 'dtype') else 'N/A'}")
                # Fall back to test pattern
                logger.warning("Falling back to test pattern due to texture creation failure")
                input_texture = self._create_test_pattern()
        else:
            # Text mode - use test pattern
            input_texture = self._create_test_pattern()

        # Create black texture for unused channels
        black_texture = self._create_black_texture()
        ref_texture = self.reference_texture if self.reference_texture else black_texture

        # Calculate timing
        current_time = time.time()
        elapsed_time = (current_time - self.start_time) * time_scale
        time_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Set uniforms
        if self.program is not None:
            try:
                self.program['iResolution'].value = (self.width, self.height, 1.0)
            except KeyError:
                pass
            try:
                self.program['iTime'].value = elapsed_time
            except KeyError:
                pass
            try:
                self.program['iTimeDelta'].value = time_delta
            except KeyError:
                pass
            try:
                self.program['iFrame'].value = self.frame_count
            except KeyError:
                pass
            try:
                self.program['iMouse'].value = (
                    mouse_x * self.width,
                    mouse_y * self.height,
                    0.0,
                    0.0
                )
            except KeyError:
                pass
            try:
                self.program['iParam1'].value = param1
            except KeyError:
                pass
            try:
                self.program['iParam2'].value = param2
            except KeyError:
                pass
            try:
                self.program['iParam3'].value = param3
            except KeyError:
                pass

            # Bind textures
            try:
                input_texture.use(location=0)
                self.program['iChannel0'].value = 0
            except KeyError:
                pass
            try:
                ref_texture.use(location=1)
                self.program['iChannel1'].value = 1
            except KeyError:
                pass
            try:
                black_texture.use(location=2)
                self.program['iChannel2'].value = 2
            except KeyError:
                pass
            try:
                black_texture.use(location=3)
                self.program['iChannel3'].value = 3
            except KeyError:
                pass

        # Render to framebuffer
        self.fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        if self.vao is not None:
            self.vao.render(moderngl.TRIANGLE_STRIP)

        # Read back result
        data = self.fbo.read(components=4)
        result = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)

        # Convert to RGB and normalize to [0, 1]
        result_rgb = result[:, :, :3].astype(np.float32) / 255.0

        # Flip vertically (OpenGL has origin at bottom-left)
        result_rgb = np.flip(result_rgb, axis=0).copy()

        # Convert to tensor with shape (1, H, W, 3)
        output = torch.from_numpy(result_rgb).unsqueeze(0)

        # Cleanup temporary textures
        input_texture.release()
        black_texture.release()

        # Increment frame counter
        self.frame_count += 1

        return {"video": output}

    def __del__(self):
        """Clean up OpenGL resources."""
        try:
            if hasattr(self, 'channel_textures'):
                for tex in self.channel_textures:
                    if tex is not None:
                        tex.release()
            if hasattr(self, 'reference_texture') and self.reference_texture is not None:
                self.reference_texture.release()
            if hasattr(self, 'output_texture') and self.output_texture is not None:
                self.output_texture.release()
            if hasattr(self, 'fbo') and self.fbo is not None:
                self.fbo.release()
            if hasattr(self, 'vao') and self.vao is not None:
                self.vao.release()
            if hasattr(self, 'program') and self.program is not None:
                self.program.release()
            if hasattr(self, 'vbo') and self.vbo is not None:
                self.vbo.release()
            if hasattr(self, 'ctx') and self.ctx is not None:
                self.ctx.release()
        except Exception:
            pass
