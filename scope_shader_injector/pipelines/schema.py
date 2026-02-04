"""Configuration schema for the Shader Injector pipeline."""

from pydantic import Field

from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


# Default shader that passes through video with optional reference blend
DEFAULT_SHADER = """void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord / iResolution.xy;
    
    // Sample video texture (Channel 0)
    vec4 video = texture(iChannel0, uv);
    
    // Output the video as-is (passthrough)
    fragColor = video;
}
"""


class ShaderInjectorConfig(BasePipelineConfig):
    """Configuration for the Shader Injector pipeline."""

    pipeline_id = "shader-injector"
    pipeline_name = "Shader Injector"
    pipeline_description = (
        "Real-time shader injection using Shadertoy-style GLSL. "
        "Write custom fragment shaders to process video frames. "
        "Supports iChannel0 (video input), iChannel1 (reference image), "
        "iResolution, iTime, iFrame, and iMouse uniforms."
    )
    pipeline_version = "0.1.0"

    # This pipeline supports prompts (shader code)
    supports_prompts = True

    # Video mode is the primary mode, but text mode allows shader preview
    modes = {
        "video": ModeDefaults(default=True),
        "text": ModeDefaults(),
    }

    # Shader code input
    shader_code: str = Field(
        default=DEFAULT_SHADER,
        description=(
            "GLSL fragment shader code in Shadertoy style. "
            "Available uniforms: iResolution, iTime, iFrame, iMouse. "
            "Available textures: iChannel0 (video), iChannel1 (reference image)."
        ),
        json_schema_extra=ui_field_config(
            order=1,
            label="Shader Code",
            category="input",
        ),
    )

    # Reference image path (optional)
    reference_image: str = Field(
        default="",
        description=(
            "Path to a reference image to use as iChannel1. "
            "Leave empty to use a black texture."
        ),
        json_schema_extra=ui_field_config(
            order=2,
            label="Reference Image",
            component="image",
            category="input",
        ),
    )

    # Time scale for iTime uniform
    time_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Multiplier for iTime uniform speed",
        json_schema_extra=ui_field_config(
            order=3,
            label="Time Scale",
        ),
    )

    # Mouse position (normalized 0-1)
    mouse_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mouse X position (0-1)",
        json_schema_extra=ui_field_config(
            order=4,
            label="Mouse X",
        ),
    )

    mouse_y: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mouse Y position (0-1)",
        json_schema_extra=ui_field_config(
            order=5,
            label="Mouse Y",
        ),
    )

    # Custom float parameters for shader experimentation
    param1: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Custom parameter 1 (available as iParam1 in shader)",
        json_schema_extra=ui_field_config(
            order=6,
            label="Param 1",
        ),
    )

    param2: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Custom parameter 2 (available as iParam2 in shader)",
        json_schema_extra=ui_field_config(
            order=7,
            label="Param 2",
        ),
    )

    param3: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Custom parameter 3 (available as iParam3 in shader)",
        json_schema_extra=ui_field_config(
            order=8,
            label="Param 3",
        ),
    )
