"""Shader Injector pipeline implementation."""

import logging
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


def create_headless_context() -> moderngl.Context:
    """Create a ModernGL context that works in headless environments.

    Tries multiple backends in order of preference:
    1. EGL (works headless on Linux with proper drivers)
    2. Default standalone (works on macOS, Windows, Linux with display)

    Returns:
        ModernGL context.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    errors = []

    # On Linux, try EGL first for headless support
    if sys.platform.startswith('linux'):
        try:
            # Try EGL backend (headless-compatible)
            # Use create_context with standalone=True and backend='egl'
            ctx = moderngl.create_context(standalone=True, backend='egl')
            logger.info("Created ModernGL context using EGL backend")
            return ctx
        except Exception as e:
            errors.append(f"EGL: {e}")
            logger.debug(f"EGL backend failed: {e}")

    # Try default standalone context
    # On macOS this uses CGL, on Windows uses WGL
    try:
        ctx = moderngl.create_context(standalone=True)
        logger.info("Created ModernGL context using default backend")
        return ctx
    except Exception as e:
        errors.append(f"Default: {e}")
        logger.debug(f"Default backend failed: {e}")

    # All backends failed
    raise RuntimeError(
        f"Failed to create OpenGL context. Tried backends: {'; '.join(errors)}. "
        "For headless Linux, ensure you have EGL support installed "
        "(e.g., libegl1-mesa-dev, or NVIDIA EGL libraries)."
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

        # Create standalone OpenGL context (headless-compatible)
        self.ctx = create_headless_context()

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

        self.output_texture = self.ctx.texture((self.width, self.height), 4)
        self.fbo = self.ctx.framebuffer(color_attachments=[self.output_texture])

    def _create_texture_from_tensor(self, tensor: torch.Tensor) -> moderngl.Texture:
        """Create a ModernGL texture from a PyTorch tensor.

        Args:
            tensor: Input tensor of shape (1, H, W, C) or (H, W, C) with values in [0, 255].

        Returns:
            ModernGL texture.
        """
        # Handle different tensor shapes
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # Convert to numpy and ensure correct dtype
        data = tensor.cpu().numpy().astype(np.uint8)
        h, w, c = data.shape

        # Ensure RGBA format
        if c == 3:
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = data
            rgba[:, :, 3] = 255
            data = rgba

        # Create texture
        texture = self.ctx.texture((w, h), 4, data.tobytes())
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False

        return texture

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
            input_texture = self._create_texture_from_tensor(video[0])
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
