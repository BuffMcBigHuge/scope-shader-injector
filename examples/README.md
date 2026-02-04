# Shader Injector Examples

This directory contains example shaders you can use with the Scope Shader Injector plugin.

## Available Uniforms

| Uniform | Type | Description |
|---------|------|-------------|
| `iResolution` | `vec3` | Viewport resolution (width, height, 1.0) |
| `iTime` | `float` | Time since start in seconds |
| `iTimeDelta` | `float` | Time since last frame |
| `iFrame` | `int` | Current frame number |
| `iMouse` | `vec4` | Mouse position (x, y, click_x, click_y) |
| `iChannel0` | `sampler2D` | Video input texture |
| `iChannel1` | `sampler2D` | Reference image texture |
| `iChannel2` | `sampler2D` | Unused (black) |
| `iChannel3` | `sampler2D` | Unused (black) |
| `iParam1` | `float` | Custom parameter 1 (0-1) |
| `iParam2` | `float` | Custom parameter 2 (0-1) |
| `iParam3` | `float` | Custom parameter 3 (0-1) |

## Shader Format

Shaders follow the Shadertoy format with a `mainImage` function:

```glsl
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 color = texture(iChannel0, uv);
    fragColor = color;
}
```

## Example Categories

### Basic Effects
- `passthrough.glsl` - Simple pass-through (no modification)
- `invert.glsl` - Invert colors
- `grayscale.glsl` - Convert to grayscale
- `brightness_contrast.glsl` - Adjust brightness and contrast

### Color Manipulation
- `hue_shift.glsl` - Shift hue over time
- `color_cycle.glsl` - Cycle through color channels
- `sepia.glsl` - Sepia tone effect
- `posterize.glsl` - Reduce color levels

### Distortion Effects
- `wave_distortion.glsl` - Animated wave distortion
- `pixelate.glsl` - Pixelation effect
- `barrel_distortion.glsl` - Lens barrel distortion
- `kaleidoscope.glsl` - Mirror kaleidoscope effect

### Visual Effects
- `vignette.glsl` - Dark vignette effect
- `chromatic_aberration.glsl` - RGB channel split
- `edge_detection.glsl` - Sobel edge detection
- `scan_lines.glsl` - CRT scanline effect

### Reference Image Effects
- `blend_reference.glsl` - Blend video with reference image
- `mask_with_reference.glsl` - Use reference as mask
- `difference.glsl` - Show difference from reference

### Advanced
- `glitch.glsl` - Digital glitch effect
- `rain.glsl` - Rain/water droplets overlay
- `matrix.glsl` - Matrix-style digital rain
- `fractal_overlay.glsl` - Mandelbrot fractal blend

## Tips

1. **Use `iParam1`, `iParam2`, `iParam3`** for tweakable parameters in real-time
2. **Use `iTime`** for animations - multiply by different values for speed control
3. **Use `iMouse`** for interactive position-based effects
4. **Combine effects** by chaining texture samples
5. **Keep shaders simple** for better performance in real-time video
