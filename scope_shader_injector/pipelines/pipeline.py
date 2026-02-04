"""Shader Injector pipeline implementation."""

import logging
import os
import sys
import threading
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

# GPU Optimization: Try to import CUDA-GL interop libraries
# These enable zero-copy transfers between CUDA and OpenGL
_CUDA_GL_AVAILABLE = False
_cuda_gl_error = None

try:
    # Try cupy first (more common, easier to install)
    import cupy as cp
    from cupy.cuda import runtime as cuda_runtime
    _CUDA_GL_AVAILABLE = True
    _CUDA_GL_BACKEND = "cupy"
    logger.info("CUDA-GL interop available via CuPy")
except ImportError as e:
    _cuda_gl_error = str(e)
    try:
        # Fall back to pycuda
        import pycuda.driver as cuda_driver
        import pycuda.gl as cuda_gl
        _CUDA_GL_AVAILABLE = True
        _CUDA_GL_BACKEND = "pycuda"
        logger.info("CUDA-GL interop available via PyCUDA")
    except ImportError as e2:
        _cuda_gl_error = f"cupy: {_cuda_gl_error}, pycuda: {e2}"
        logger.debug(f"CUDA-GL interop not available: {_cuda_gl_error}")


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
        logger.info(f"Init thread ID: {threading.get_ident()} (OpenGL will be initialized lazily)")
        
        # Store configuration from kwargs (passed at pipeline load time)
        self.initial_shader_code = kwargs.get("shader_code", DEFAULT_SHADER)
        self.initial_reference_image = kwargs.get("reference_image", "")
        self.initial_time_scale = kwargs.get("time_scale", 1.0)
        self.initial_param1 = kwargs.get("param1", 0.5)
        self.initial_param2 = kwargs.get("param2", 0.5)
        self.initial_param3 = kwargs.get("param3", 0.5)
        
        logger.info(f"Initial shader code length: {len(self.initial_shader_code)} chars")
        
        # OpenGL resources - will be initialized lazily in the worker thread
        self._gl_initialized = False
        self._gl_thread_id = None
        self.ctx = None
        self.vbo = None
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
        
        # GPU Optimization: Texture pool for reuse (avoids per-frame allocation)
        self._texture_pool: dict[tuple[int, int], moderngl.Texture] = {}
        self._persistent_black_texture: moderngl.Texture | None = None
        self._persistent_test_pattern: moderngl.Texture | None = None
        self._test_pattern_size: tuple[int, int] | None = None
        
        # GPU Optimization: PBO buffers for async transfers
        self._upload_pbo: moderngl.Buffer | None = None
        self._download_pbo: moderngl.Buffer | None = None
        self._pbo_size: int = 0
        
        # Track input tensor device for CUDA output optimization
        self._input_device: torch.device | None = None
        
        # CUDA-GL interop state
        self._cuda_gl_enabled = False
        self._cuda_gl_checked = False

    def _check_cuda_gl_interop(self) -> bool:
        """Check if CUDA-GL interop can be used.
        
        This checks both library availability and runtime compatibility.
        
        Returns:
            True if CUDA-GL interop is available and working.
        """
        if self._cuda_gl_checked:
            return self._cuda_gl_enabled
        
        self._cuda_gl_checked = True
        
        if not _CUDA_GL_AVAILABLE:
            logger.info(f"CUDA-GL interop not available: {_cuda_gl_error}")
            return False
        
        if not torch.cuda.is_available():
            logger.info("CUDA-GL interop disabled: CUDA not available")
            return False
        
        # Check if OpenGL context is compatible
        try:
            vendor = self.ctx.info.get('GL_VENDOR', '').lower()
            if 'nvidia' not in vendor:
                logger.info(f"CUDA-GL interop disabled: Non-NVIDIA GPU ({vendor})")
                return False
            
            self._cuda_gl_enabled = True
            logger.info("CUDA-GL interop enabled for zero-copy transfers")
            return True
            
        except Exception as e:
            logger.warning(f"CUDA-GL interop check failed: {e}")
            return False

    def _initialize_gl(self):
        """Initialize OpenGL resources. Called lazily from the worker thread."""
        if self._gl_initialized:
            return
        
        current_thread = threading.get_ident()
        logger.info(f"Initializing OpenGL in worker thread: {current_thread}")
        
        # Create standalone OpenGL context (headless-compatible)
        self.ctx = create_headless_context()
        self._gl_thread_id = current_thread
        
        # Log context info
        logger.info(f"OpenGL Version: {self.ctx.version_code}")
        logger.info(f"OpenGL Vendor: {self.ctx.info.get('GL_VENDOR', 'Unknown')}")
        logger.info(f"OpenGL Renderer: {self.ctx.info.get('GL_RENDERER', 'Unknown')}")
        logger.info(f"Max Texture Size: {self.ctx.info.get('GL_MAX_TEXTURE_SIZE', 'Unknown')}")
        
        # Test basic texture creation to verify context is working
        try:
            test_data = np.zeros((16, 16, 4), dtype=np.uint8)
            test_texture = self.ctx.texture((16, 16), 4, test_data.tobytes())
            test_texture.release()
            logger.info("OpenGL context texture test passed")
        except Exception as e:
            logger.error(f"OpenGL context texture test FAILED: {e}")
            raise RuntimeError(f"OpenGL context is not functional: {e}")

        # Create fullscreen quad geometry
        vertices = np.array([
            # position    texcoord
            -1.0, -1.0,   0.0, 0.0,
            1.0, -1.0,    1.0, 0.0,
            -1.0, 1.0,    0.0, 1.0,
            1.0, 1.0,     1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)

        # Mark as initialized BEFORE calling methods that might check this
        self._gl_initialized = True
        
        # Compile the initial shader (from config, not default)
        logger.info(f"Compiling initial shader ({len(self.initial_shader_code)} chars)")
        self._compile_shader(self.initial_shader_code)
        self._setup_framebuffer()
        
        logger.info("OpenGL initialization complete")

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        """Return the configuration class for this pipeline."""
        return ShaderInjectorConfig

    def _preprocess_shader_code(self, shader_code: str) -> str:
        """Preprocess shader code to handle flattened single-line input.
        
        When shader code is passed through certain interfaces, newlines may be
        removed, causing // comments to comment out the entire shader.
        This function attempts to restore proper line breaks.
        
        Args:
            shader_code: Raw shader code, possibly flattened to single line.
            
        Returns:
            Shader code with proper line breaks restored.
        """
        import re
        
        # Step 0: Sanitize input - remove characters that GLSL doesn't support
        # This handles JSON escaping artifacts and copy-paste issues
        
        # Remove any quote characters (GLSL doesn't support strings)
        if '"' in shader_code or "'" in shader_code:
            logger.warning("Removing quote characters from shader code (GLSL doesn't support strings)")
            shader_code = shader_code.replace('"', '').replace("'", '')
        
        # Rename GLSL reserved words that are commonly used as variable names
        # 'char' is reserved in GLSL but people often use it for character-related code
        glsl_reserved_renames = {
            r'\bchar\b': 'charVar',      # char -> charVar
            r'\bclass\b': 'classVar',    # class -> classVar  
            r'\btemplate\b': 'templateVar',  # template -> templateVar
        }
        for pattern, replacement in glsl_reserved_renames.items():
            if re.search(pattern, shader_code):
                logger.warning(f"Renaming GLSL reserved word in shader code: {pattern} -> {replacement}")
                shader_code = re.sub(pattern, replacement, shader_code)
        
        # Remove common JSON escape sequences that might have leaked through
        shader_code = shader_code.replace('\\n', '\n')  # Convert literal \n to newline
        shader_code = shader_code.replace('\\t', ' ')   # Convert literal \t to space
        shader_code = shader_code.replace('\\/', '/')   # Unescape forward slash
        
        # If the code already has proper newlines, return as-is
        if shader_code.count('\n') > 3:
            return shader_code
        
        # If no mainImage, nothing to do
        if 'mainImage' not in shader_code:
            return shader_code
        
        logger.info(f"Preprocessing shader code ({len(shader_code)} chars, {shader_code.count(chr(10))} newlines)")
        
        # Step 1: Extract everything before 'void mainImage' as comments
        # and everything from 'void mainImage' onwards as the actual code
        main_idx = shader_code.find('void mainImage')
        
        # Also check for truncated input like 'mainImage(' without 'void'
        if main_idx == -1:
            main_idx = shader_code.find('mainImage(')
            if main_idx != -1:
                # Insert 'void ' before mainImage
                shader_code = shader_code[:main_idx] + 'void ' + shader_code[main_idx:]
                main_idx = shader_code.find('void mainImage')
                logger.warning("Shader was missing 'void' before mainImage, auto-fixed")
        
        # Check for even more truncated input 'Image(' at the start
        if main_idx == -1:
            if shader_code.strip().startswith('Image('):
                shader_code = 'void main' + shader_code.strip()
                main_idx = 0
                logger.warning("Shader was truncated to 'Image(', auto-fixed to 'void mainImage('")
        
        if main_idx == -1:
            logger.warning("Could not find mainImage function in shader code")
            return shader_code
        
        # Everything before mainImage - could be comments AND helper functions
        before_main = shader_code[:main_idx].strip()
        code_part = shader_code[main_idx:]
        
        # Separate comments from helper functions in the before_main part
        # Helper functions contain patterns like "float funcname(" or "vec3 funcname("
        comments_part = ""
        helper_functions_part = ""
        
        if before_main:
            # Find where actual code (helper functions) starts
            # Look for function definitions: type name(
            func_match = re.search(r'(float|int|vec[234]|mat[234]|void|bool)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', before_main)
            
            if func_match:
                # There are helper functions - split comments from functions
                func_start = func_match.start()
                comments_part = before_main[:func_start].strip()
                helper_functions_part = before_main[func_start:]
                logger.info(f"Found helper functions before mainImage: {helper_functions_part[:60]}...")
            else:
                # No helper functions, everything is comments
                comments_part = before_main
        
        # Step 2: Format the code part properly
        # Replace multiple spaces with single space first
        code_part = re.sub(r'  +', ' ', code_part)
        
        # Add newlines at key positions
        # After opening brace
        code_part = re.sub(r'\{\s*', '{\n    ', code_part)
        # Before closing brace
        code_part = re.sub(r'\s*\}', '\n}', code_part)
        # After semicolons followed by code OR comments
        code_part = re.sub(r';\s*(?=[a-zA-Z/])', ';\n    ', code_part)
        
        # Step 3: Handle any // comments within the code
        # They need to end at the next statement
        lines = code_part.split('\n')
        fixed_lines = []
        for line in lines:
            line = line.strip()
            if '//' in line:
                # Find if there's code after the comment
                comment_idx = line.find('//')
                before_comment = line[:comment_idx].strip()
                after_comment = line[comment_idx:]
                
                # Check if the comment has code after it (after the comment text)
                # Pattern: // some text followed by a type keyword starting a statement
                # Match common GLSL types (must be followed by identifier or opening paren)
                # Note: Use .*? instead of [^/]*? because comments may contain / (e.g., "barrel/pincushion")
                match = re.search(
                    r'//.*?\s+(void\s+\w|vec[234]\s+\w|mat[234]\s+\w|float\s+\w|int\s+\w|bool\s+\w|fragColor\s*=|return\s)', 
                    after_comment
                )
                # Control flow (if, for, while followed by (, or else)
                if not match:
                    match = re.search(r'//.*?\s+(if\s*\(|for\s*\(|while\s*\(|else\s)', after_comment)
                # Variable assignments including compound assignments (+=, -=, *=, /=)
                if not match:
                    match = re.search(r'//.*?\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[+\-*/]?=', after_comment)
                
                if match:
                    # Split the comment from the code
                    split_pos = match.start(1)
                    comment_text = after_comment[:split_pos].rstrip()
                    code_after = after_comment[split_pos:]
                    if before_comment:
                        fixed_lines.append(before_comment)
                    fixed_lines.append(comment_text)
                    fixed_lines.append('    ' + code_after)
                else:
                    fixed_lines.append(line)
            else:
                if line:
                    fixed_lines.append(line)
        
        # Step 4: Combine comments, helper functions, and formatted main code
        result_lines = []
        
        # Add comments first
        if comments_part:
            # Split comments by // to put each on its own line
            comment_parts = comments_part.split('//')
            for i, part in enumerate(comment_parts):
                part = part.strip()
                if part:
                    if i > 0:
                        result_lines.append('// ' + part)
                    else:
                        # First part might not have // prefix if it's leading text
                        if comment_parts[0].strip():
                            result_lines.append(part)
        
        # Add helper functions (need formatting too)
        if helper_functions_part:
            # Format helper functions similar to main code
            helpers = helper_functions_part
            helpers = re.sub(r'  +', ' ', helpers)
            helpers = re.sub(r'\{\s*', '{\n    ', helpers)
            helpers = re.sub(r'\s*\}', '\n}', helpers)
            helpers = re.sub(r';\s*(?=[a-zA-Z/])', ';\n    ', helpers)
            
            # Apply the same inline comment handling as main code
            helper_lines = helpers.split('\n')
            for line in helper_lines:
                line = line.strip()
                if not line:
                    continue
                    
                if '//' in line:
                    comment_idx = line.find('//')
                    before_comment = line[:comment_idx].strip()
                    after_comment = line[comment_idx:]
                    
                    # Check for code after comment
                    match = re.search(
                        r'//.*?\s+(void\s+\w|vec[234]\s+\w|mat[234]\s+\w|float\s+\w|int\s+\w|bool\s+\w|return\s)', 
                        after_comment
                    )
                    if not match:
                        match = re.search(r'//.*?\s+(if\s*\(|for\s*\(|while\s*\(|else\s)', after_comment)
                    if not match:
                        match = re.search(r'//.*?\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[+\-*/]?=', after_comment)
                    
                    if match:
                        split_pos = match.start(1)
                        comment_text = after_comment[:split_pos].rstrip()
                        code_after = after_comment[split_pos:]
                        if before_comment:
                            result_lines.append(before_comment)
                        result_lines.append(comment_text)
                        result_lines.append('    ' + code_after)
                    else:
                        result_lines.append(line)
                else:
                    result_lines.append(line)
            
            # Add blank line before mainImage
            result_lines.append('')
        
        # Add the main function code
        result_lines.extend(fixed_lines)
        
        result = '\n'.join(result_lines)
        
        # Clean up any double newlines
        result = re.sub(r'\n\s*\n', '\n', result)
        
        logger.info(f"Preprocessed shader:\n{result}")
        return result

    def _compile_shader(self, shader_code: str) -> bool:
        """Compile a Shadertoy-style shader.

        Args:
            shader_code: User's GLSL shader code with mainImage function.

        Returns:
            True if compilation succeeded, False otherwise.
        """
        if shader_code == self.current_shader_code and self.program is not None:
            return True

        # Preprocess shader code to handle flattened input
        shader_code = self._preprocess_shader_code(shader_code)

        # Validate shader code contains mainImage function
        if "mainImage" not in shader_code:
            logger.warning("Shader code missing 'mainImage' function, using default shader")
            if shader_code != DEFAULT_SHADER:
                return self._compile_shader(DEFAULT_SHADER)
            return False

        # Build full fragment shader
        fragment_shader = FRAGMENT_SHADER_TEMPLATE.format(user_code=shader_code)
        
        # Log the shader being compiled for debugging
        logger.debug(f"Compiling fragment shader:\n{fragment_shader[:500]}...")

        # Store old program/vao in case we need to restore
        old_program = self.program
        old_vao = self.vao
        old_shader_code = self.current_shader_code

        try:
            # Compile new program
            logger.info("Compiling shader program...")
            new_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader=fragment_shader,
            )
            logger.info("Shader program compiled successfully")

            # Create VAO - always use skip_errors=True to handle optimized-out attributes
            # Per ModernGL docs, attributes not used in shader may be optimized out
            new_vao = self.ctx.vertex_array(
                new_program,
                [(self.vbo, '2f 2f', 'in_position', 'in_texcoord')],
                skip_errors=True,
            )

            # Success - clean up old resources and use new ones
            if old_program is not None:
                old_program.release()
            if old_vao is not None:
                old_vao.release()

            self.program = new_program
            self.vao = new_vao
            self.current_shader_code = shader_code
            logger.info(f"Shader compiled successfully: {shader_code[:80]}...")
            return True

        except Exception as e:
            logger.error(f"Shader compilation error: {e}")
            
            # Keep the old working program if we have one
            if old_program is not None:
                logger.info("Keeping previous working shader")
                self.program = old_program
                self.vao = old_vao
                self.current_shader_code = old_shader_code
                return False
            
            # No previous shader, try default
            if shader_code != DEFAULT_SHADER:
                logger.info("Falling back to default shader")
                return self._compile_shader(DEFAULT_SHADER)
            
            # Even default shader failed - this is a serious error
            logger.error("Default shader compilation failed - OpenGL context may be broken")
            return False

    def _ensure_context_active(self):
        """Ensure the OpenGL context is active/current.
        
        For standalone contexts, this makes the context current for the calling thread.
        This also initializes OpenGL if not yet done (lazy initialization).
        """
        # Initialize OpenGL if not yet done
        if not self._gl_initialized:
            self._initialize_gl()
            return  # Context is already current after initialization
        
        # Check if we're in the same thread
        current_thread = threading.get_ident()
        if self._gl_thread_id and current_thread != self._gl_thread_id:
            logger.error(f"OpenGL context used from wrong thread! GL thread: {self._gl_thread_id}, Current: {current_thread}")
            raise RuntimeError("OpenGL context cannot be used from a different thread")
        
        try:
            # For standalone contexts, we need to make sure it's current
            if hasattr(self.ctx, 'mglo') and hasattr(self.ctx.mglo, 'make_current'):
                self.ctx.mglo.make_current()
            else:
                self.ctx.__enter__()
        except Exception as e:
            logger.debug(f"Context activation: {e}")
            # Context might already be current, which is fine

    def _setup_framebuffer(self):
        """Set up the output framebuffer."""
        if self.output_texture is not None:
            self.output_texture.release()
        if self.fbo is not None:
            self.fbo.release()

        logger.info(f"Setting up framebuffer: {self.width}x{self.height}")
        
        # Ensure context is active
        self._ensure_context_active()
        
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

    def _get_or_create_texture(self, width: int, height: int) -> moderngl.Texture:
        """Get a texture from the pool or create a new one.
        
        GPU Optimization: Reuses textures instead of creating new ones each frame.
        This eliminates texture allocation overhead which is significant.
        
        Args:
            width: Texture width.
            height: Texture height.
            
        Returns:
            ModernGL texture (may contain old data, caller must write new data).
        """
        key = (width, height)
        if key not in self._texture_pool:
            logger.debug(f"Creating new pooled texture: {width}x{height}")
            texture = self.ctx.texture((width, height), 4)
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = False
            texture.repeat_y = False
            self._texture_pool[key] = texture
        return self._texture_pool[key]
    
    def _get_upload_pbo(self, size: int) -> moderngl.Buffer:
        """Get or create a PBO buffer for texture uploads.
        
        GPU Optimization: PBOs enable DMA transfers that don't block the CPU.
        The buffer is reused across frames to avoid allocation overhead.
        
        Args:
            size: Required buffer size in bytes.
            
        Returns:
            ModernGL buffer suitable for texture upload.
        """
        if self._upload_pbo is None or self._pbo_size < size:
            if self._upload_pbo is not None:
                self._upload_pbo.release()
            
            # Allocate with some headroom to avoid frequent reallocations
            alloc_size = max(size, self.width * self.height * 4)
            logger.debug(f"Creating upload PBO: {alloc_size} bytes")
            
            # Use dynamic=True for frequently updated buffers
            self._upload_pbo = self.ctx.buffer(reserve=alloc_size, dynamic=True)
            self._pbo_size = alloc_size
        
        return self._upload_pbo
    
    def _upload_texture_via_pbo(self, texture: moderngl.Texture, data: bytes, width: int, height: int):
        """Upload texture data using a PBO for better performance.
        
        GPU Optimization: Using a PBO allows the GPU to perform DMA transfers
        asynchronously, reducing CPU stalls during texture uploads.
        
        Args:
            texture: Target texture.
            data: Raw pixel data (RGBA, uint8).
            width: Texture width.
            height: Texture height.
        """
        size = len(data)
        pbo = self._get_upload_pbo(size)
        
        # Write data to PBO (this can be done while GPU is busy)
        pbo.write(data)
        
        # Copy from PBO to texture (GPU handles this efficiently)
        texture.write(pbo)
    
    def _get_download_pbo(self, size: int) -> moderngl.Buffer:
        """Get or create a PBO buffer for framebuffer readback.
        
        GPU Optimization: PBOs enable async readback that doesn't block rendering.
        
        Args:
            size: Required buffer size in bytes.
            
        Returns:
            ModernGL buffer suitable for framebuffer readback.
        """
        if self._download_pbo is None or self._download_pbo.size < size:
            if self._download_pbo is not None:
                self._download_pbo.release()
            
            logger.debug(f"Creating download PBO: {size} bytes")
            # Reserve buffer for readback
            self._download_pbo = self.ctx.buffer(reserve=size)
        
        return self._download_pbo
    
    def _read_framebuffer_via_pbo(self) -> bytes:
        """Read framebuffer data using a PBO for better performance.
        
        GPU Optimization: Using a PBO for readback allows the GPU to transfer
        data asynchronously, reducing CPU stalls.
        
        Returns:
            Raw pixel data (RGBA, uint8).
        """
        size = self.width * self.height * 4
        pbo = self._get_download_pbo(size)
        
        # Read framebuffer into PBO (GPU initiates DMA transfer)
        self.fbo.read_into(pbo, components=4)
        
        # Read from PBO to CPU (may still block, but transfer is optimized)
        return pbo.read()
    
    def _try_cuda_tensor_upload(self, tensor: torch.Tensor, texture: moderngl.Texture) -> bool:
        """Try to upload tensor data using CUDA-GL interop (zero-copy).
        
        GPU Optimization: When available, this enables direct GPU-to-GPU transfer
        without any CPU involvement, providing maximum performance.
        
        Args:
            tensor: CUDA tensor to upload.
            texture: Target OpenGL texture.
            
        Returns:
            True if CUDA-GL upload succeeded, False to fall back to PBO.
        """
        if not self._check_cuda_gl_interop():
            return False
        
        if not tensor.is_cuda:
            return False
        
        try:
            # This is a placeholder for full CUDA-GL interop implementation
            # Full implementation requires:
            # 1. Register OpenGL texture with CUDA (cudaGraphicsGLRegisterImage)
            # 2. Map the resource (cudaGraphicsMapResources)
            # 3. Get CUDA array from mapped resource
            # 4. Copy from CUDA tensor to CUDA array (cudaMemcpy2DToArray)
            # 5. Unmap resource (cudaGraphicsUnmapResources)
            #
            # For now, we use an optimized path that minimizes copies:
            # - Keep tensor on GPU as long as possible
            # - Use pinned memory for the CPU transfer
            
            if _CUDA_GL_AVAILABLE and _CUDA_GL_BACKEND == "cupy":
                # CuPy path: Use pinned memory for faster transfer
                # This is still faster than regular numpy conversion
                with cp.cuda.Stream() as stream:
                    # Convert torch tensor to cupy array (zero-copy on same device)
                    if tensor.dim() == 4:
                        tensor = tensor.squeeze(0)
                    
                    # Ensure contiguous
                    tensor = tensor.contiguous()
                    
                    # Get data pointer and create cupy array view
                    # Note: This shares memory with the torch tensor
                    cp_array = cp.asarray(tensor)
                    
                    # Process on GPU: normalize, convert to uint8, add alpha
                    h, w = cp_array.shape[:2]
                    c = cp_array.shape[2] if cp_array.ndim > 2 else 1
                    
                    # Normalize to [0, 255]
                    if cp_array.max() <= 1.0:
                        cp_array = cp_array * 255.0
                    
                    cp_array = cp.clip(cp_array, 0, 255).astype(cp.uint8)
                    
                    # Convert to RGBA
                    if c == 3:
                        rgba = cp.zeros((h, w, 4), dtype=cp.uint8)
                        rgba[:, :, :3] = cp_array
                        rgba[:, :, 3] = 255
                        cp_array = rgba
                    elif c == 1:
                        rgba = cp.zeros((h, w, 4), dtype=cp.uint8)
                        rgba[:, :, 0] = cp_array[:, :, 0] if cp_array.ndim > 2 else cp_array
                        rgba[:, :, 1] = rgba[:, :, 0]
                        rgba[:, :, 2] = rgba[:, :, 0]
                        rgba[:, :, 3] = 255
                        cp_array = rgba
                    
                    # Flip for OpenGL
                    cp_array = cp.flipud(cp_array)
                    
                    # Sync and get bytes (this is the only CPU transfer)
                    stream.synchronize()
                    data_bytes = cp.asnumpy(cp_array).tobytes()
                    
                    # Upload via PBO (already optimized)
                    self._upload_texture_via_pbo(texture, data_bytes, w, h)
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"CUDA-GL interop upload failed, falling back to PBO: {e}")
            return False
    
    def _prepare_tensor_data(self, tensor: torch.Tensor) -> tuple[np.ndarray, int, int]:
        """Prepare tensor data for GPU upload.
        
        Converts PyTorch tensor to numpy array in RGBA uint8 format.
        
        Args:
            tensor: Input tensor of shape (1, H, W, C) or (H, W, C).
            
        Returns:
            Tuple of (data array, width, height).
        """
        # Track input device for potential CUDA output optimization
        self._input_device = tensor.device
        
        # Handle different tensor shapes
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Log tensor info for debugging
        logger.debug(f"Input tensor shape: {tensor.shape}, dtype: {tensor.dtype}, "
                     f"device: {tensor.device}")

        # Convert to numpy (this is the CPU transfer - unavoidable without CUDA-GL interop)
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

        # Flip vertically for OpenGL (origin at bottom-left)
        data = np.flipud(data)
        
        # Ensure contiguous C-order array
        data = np.ascontiguousarray(data, dtype=np.uint8)
        
        return data, w, h

    def _create_texture_from_tensor(self, tensor: torch.Tensor) -> tuple[moderngl.Texture, bool]:
        """Create or update a ModernGL texture from a PyTorch tensor.

        GPU Optimization: Uses texture pooling to avoid per-frame allocation.
        Tries CUDA-GL interop first for zero-copy, falls back to PBO.
        Returns a tuple indicating if the texture is from the pool (should not be released).

        Args:
            tensor: Input tensor of shape (1, H, W, C) or (H, W, C).
                    Values can be in [0, 255] or [0, 1] range.

        Returns:
            Tuple of (ModernGL texture, is_pooled).
            If is_pooled is True, caller should NOT release the texture.
        """
        # Ensure context is active before GPU operations
        self._ensure_context_active()
        
        # Track input device for CUDA output optimization
        self._input_device = tensor.device
        
        # Get dimensions for texture pool lookup
        if tensor.dim() == 4:
            _, h, w, c = tensor.shape
        elif tensor.dim() == 3:
            h, w, c = tensor.shape
        else:
            h, w = tensor.shape
            c = 1
        
        # Validate dimensions
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid texture dimensions: {w}x{h}")
        
        try:
            # GPU Optimization: Reuse texture from pool
            texture = self._get_or_create_texture(w, h)
            
            # GPU Optimization: Try CUDA-GL interop first (zero-copy)
            if tensor.is_cuda and self._try_cuda_tensor_upload(tensor, texture):
                logger.debug(f"Used CUDA-GL interop for texture upload: {w}x{h}")
                return texture, True
            
            # Fall back to PBO-based upload
            data, w, h = self._prepare_tensor_data(tensor)
            
            # Validate data size
            expected_size = w * h * 4
            actual_size = data.nbytes
            if actual_size != expected_size:
                raise ValueError(f"Data size mismatch: expected {expected_size}, got {actual_size}")
            
            logger.debug(f"Updating texture via PBO: {w}x{h}, data size: {actual_size}")
            
            # GPU Optimization: Use PBO for async upload
            data_bytes = data.tobytes()
            self._upload_texture_via_pbo(texture, data_bytes, w, h)
            
            return texture, True  # True = pooled, don't release
            
        except Exception as e:
            logger.error(f"Failed to update texture: {e}")
            logger.error(f"Texture params: size=({w}, {h}), components=4")
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
            # Ensure context is active
            self._ensure_context_active()
            
            img = Image.open(path).convert('RGBA')
            img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            data = np.array(img, dtype=np.uint8)
            data = np.ascontiguousarray(data)

            texture = self.ctx.texture((self.width, self.height), 4, data.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            return texture

        except Exception as e:
            logger.error(f"Failed to load reference image: {e}")
            return None

    def _get_black_texture(self) -> moderngl.Texture:
        """Get or create a persistent black texture for unused channels.
        
        GPU Optimization: Reuses a single black texture instead of creating per frame.
        """
        # Ensure context is active
        self._ensure_context_active()
        
        # Check if we need to create or recreate the texture
        if self._persistent_black_texture is None:
            logger.debug(f"Creating persistent black texture: {self.width}x{self.height}")
            data = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            data[:, :, 3] = 255  # Full alpha
            data = np.ascontiguousarray(data)

            self._persistent_black_texture = self.ctx.texture(
                (self.width, self.height), 4, data.tobytes()
            )
            self._persistent_black_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        return self._persistent_black_texture

    def _get_test_pattern(self) -> moderngl.Texture:
        """Get or create a persistent test pattern texture for text mode.
        
        GPU Optimization: Reuses test pattern texture instead of creating per frame.
        """
        # Ensure context is active
        self._ensure_context_active()
        
        current_size = (self.width, self.height)
        
        # Check if we need to create or recreate the texture
        if (self._persistent_test_pattern is None or 
            self._test_pattern_size != current_size):
            
            logger.debug(f"Creating persistent test pattern: {self.width}x{self.height}")
            
            # Release old texture if size changed
            if self._persistent_test_pattern is not None:
                self._persistent_test_pattern.release()
            
            # Create a colorful gradient pattern
            y, x = np.mgrid[0:self.height, 0:self.width]
            r = ((x / self.width) * 255).astype(np.uint8)
            g = ((y / self.height) * 255).astype(np.uint8)
            b = (((x + y) / (self.width + self.height)) * 255).astype(np.uint8)
            a = np.full((self.height, self.width), 255, dtype=np.uint8)

            data = np.stack([r, g, b, a], axis=-1)
            data = np.ascontiguousarray(data)
            
            self._persistent_test_pattern = self.ctx.texture(
                (self.width, self.height), 4, data.tobytes()
            )
            self._persistent_test_pattern.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._test_pattern_size = current_size
        
        return self._persistent_test_pattern

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
        # Ensure OpenGL context is active for this thread
        self._ensure_context_active()
        
        # Get shader code - prefer runtime kwargs, fall back to initial config
        shader_code = kwargs.get("shader_code", kwargs.get("prompt", None))
        
        # If no runtime shader code provided, use the initial configuration
        if shader_code is None or shader_code.strip() == "":
            shader_code = self.initial_shader_code
        
        # Validate shader code - if clearly not GLSL, use initial shader
        if "mainImage" not in shader_code:
            # Check if it looks like natural language (no GLSL keywords)
            glsl_indicators = ["void", "vec", "float", "int", "uniform", "texture", "fragColor"]
            if not any(indicator in shader_code for indicator in glsl_indicators):
                logger.debug("Input doesn't appear to be GLSL code, using initial shader")
                shader_code = self.initial_shader_code
            else:
                logger.warning("Shader code missing 'mainImage' function, using initial shader")
                shader_code = self.initial_shader_code

        # Compile shader if changed
        compile_success = self._compile_shader(shader_code)
        
        # Ensure we have a working program - critical for rendering
        if self.program is None:
            logger.error("No working shader program available")
            if not compile_success:
                # Last resort - try default shader one more time
                self._compile_shader(DEFAULT_SHADER)
            if self.program is None:
                raise RuntimeError("Failed to compile any shader - cannot render")

        # Get parameters - prefer runtime kwargs, fall back to initial config
        time_scale = kwargs.get("time_scale", self.initial_time_scale)
        mouse_x = kwargs.get("mouse_x", 0.5)
        mouse_y = kwargs.get("mouse_y", 0.5)
        param1 = kwargs.get("param1", self.initial_param1)
        param2 = kwargs.get("param2", self.initial_param2)
        param3 = kwargs.get("param3", self.initial_param3)

        # Handle reference image - prefer runtime, fall back to initial
        reference_image = kwargs.get("reference_image", self.initial_reference_image)
        if reference_image != self.reference_image_path:
            if self.reference_texture is not None:
                self.reference_texture.release()
            self.reference_texture = self._load_reference_image(reference_image)
            self.reference_image_path = reference_image

        # Get video input or create test pattern
        # GPU Optimization: Track if texture is pooled (should not be released)
        input_texture_pooled = False
        video = kwargs.get("video")
        if video is not None and len(video) > 0:
            # Video mode - use input frame
            frame = video[0]
            logger.debug(f"Processing video frame: type={type(frame)}, shape={frame.shape if hasattr(frame, 'shape') else 'N/A'}")
            try:
                input_texture, input_texture_pooled = self._create_texture_from_tensor(frame)
            except Exception as e:
                logger.error(f"Failed to create texture from video frame: {e}")
                logger.error(f"Frame info: type={type(frame)}, shape={frame.shape if hasattr(frame, 'shape') else 'N/A'}, "
                            f"dtype={frame.dtype if hasattr(frame, 'dtype') else 'N/A'}")
                # Fall back to test pattern (persistent, don't release)
                logger.warning("Falling back to test pattern due to texture creation failure")
                input_texture = self._get_test_pattern()
                input_texture_pooled = True
        else:
            # Text mode - use persistent test pattern
            input_texture = self._get_test_pattern()
            input_texture_pooled = True

        # GPU Optimization: Use persistent black texture (don't release)
        black_texture = self._get_black_texture()
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
        else:
            logger.error("VAO is None - cannot render!")

        # GPU Optimization: Read back result using PBO for async transfer
        data = self._read_framebuffer_via_pbo()
        result = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 4)

        # Convert to RGB and normalize to [0, 1]
        result_rgb = result[:, :, :3].astype(np.float32) / 255.0

        # Flip vertically (OpenGL has origin at bottom-left)
        result_rgb = np.flip(result_rgb, axis=0).copy()

        # Convert to tensor with shape (1, H, W, 3)
        output = torch.from_numpy(result_rgb).unsqueeze(0)
        
        # GPU Optimization: If input was on CUDA, move output to CUDA
        # This allows downstream operations to stay on GPU
        if self._input_device is not None and self._input_device.type == 'cuda':
            output = output.to(self._input_device)

        # Log every 100 frames to show we're processing
        if self.frame_count % 100 == 0:
            # Sample center pixel to verify rendering
            center_y, center_x = self.height // 2, self.width // 2
            center_pixel = result_rgb[center_y, center_x]
            logger.info(f"Frame {self.frame_count}: output shape={output.shape}, "
                       f"center_pixel={center_pixel}, shader={self.current_shader_code[:50] if self.current_shader_code else 'None'}...")

        # GPU Optimization: Don't release pooled/persistent textures
        # They will be reused in the next frame
        # (input_texture_pooled and black_texture are persistent)

        # Increment frame counter
        self.frame_count += 1

        return {"video": output}

    def __del__(self):
        """Clean up OpenGL resources."""
        try:
            # Clean up texture pool
            if hasattr(self, '_texture_pool'):
                for tex in self._texture_pool.values():
                    if tex is not None:
                        tex.release()
            
            # Clean up persistent textures
            if hasattr(self, '_persistent_black_texture') and self._persistent_black_texture is not None:
                self._persistent_black_texture.release()
            if hasattr(self, '_persistent_test_pattern') and self._persistent_test_pattern is not None:
                self._persistent_test_pattern.release()
            
            # Clean up PBO buffers
            if hasattr(self, '_upload_pbo') and self._upload_pbo is not None:
                self._upload_pbo.release()
            if hasattr(self, '_download_pbo') and self._download_pbo is not None:
                self._download_pbo.release()
            
            # Clean up channel textures
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
