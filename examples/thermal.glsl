// Thermal Vision - Heat map / infrared camera effect
// iParam1: Sensitivity (contrast)
// iParam2: Color palette (0 = classic, 1 = iron)

vec3 heatPalette(float t, float style) {
    // Classic thermal: black -> blue -> magenta -> red -> yellow -> white
    vec3 classic = vec3(
        smoothstep(0.4, 0.7, t),
        smoothstep(0.7, 1.0, t),
        smoothstep(0.0, 0.3, t) * (1.0 - smoothstep(0.5, 0.7, t)) + smoothstep(0.9, 1.0, t)
    );
    
    // Iron palette: black -> purple -> red -> orange -> yellow -> white
    vec3 iron = vec3(
        smoothstep(0.2, 0.5, t),
        smoothstep(0.5, 0.8, t) * 0.5 + smoothstep(0.8, 1.0, t) * 0.5,
        smoothstep(0.0, 0.2, t) * 0.7 * (1.0 - smoothstep(0.4, 0.6, t))
    );
    
    return mix(classic, iron, style);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Calculate "heat" based on luminance
    float heat = dot(video.rgb, vec3(0.299, 0.587, 0.114));
    
    // Apply sensitivity (contrast)
    float sensitivity = iParam1 * 2.0 + 0.5;  // 0.5 to 2.5
    heat = pow(heat, 1.0 / sensitivity);
    
    // Map to color palette
    vec3 color = heatPalette(heat, iParam2);
    
    fragColor = vec4(color, 1.0);
}
