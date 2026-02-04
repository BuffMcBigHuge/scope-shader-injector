// Blend Reference - Blend video with reference image
// iParam1: Blend amount (0 = video only, 1 = reference only)
// iParam2: Blend mode (0 = mix, 0.33 = multiply, 0.66 = screen, 1 = overlay)

vec3 blendMultiply(vec3 base, vec3 blend) {
    return base * blend;
}

vec3 blendScreen(vec3 base, vec3 blend) {
    return 1.0 - (1.0 - base) * (1.0 - blend);
}

vec3 blendOverlay(vec3 base, vec3 blend) {
    return mix(
        2.0 * base * blend,
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend),
        step(0.5, base)
    );
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    vec4 video = texture(iChannel0, uv);
    vec4 ref = texture(iChannel1, uv);
    
    float amount = iParam1;
    float mode = iParam2;
    
    vec3 result;
    
    if (mode < 0.25) {
        // Simple mix
        result = mix(video.rgb, ref.rgb, amount);
    } else if (mode < 0.5) {
        // Multiply blend
        vec3 blended = blendMultiply(video.rgb, ref.rgb);
        result = mix(video.rgb, blended, amount);
    } else if (mode < 0.75) {
        // Screen blend
        vec3 blended = blendScreen(video.rgb, ref.rgb);
        result = mix(video.rgb, blended, amount);
    } else {
        // Overlay blend
        vec3 blended = blendOverlay(video.rgb, ref.rgb);
        result = mix(video.rgb, blended, amount);
    }
    
    fragColor = vec4(result, 1.0);
}
