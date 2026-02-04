// Vignette - Dark corners effect
// iParam1: Vignette intensity
// iParam2: Vignette radius/softness

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Calculate distance from center
    vec2 centered = uv - 0.5;
    float dist = length(centered);
    
    // Vignette parameters
    float intensity = iParam1;
    float radius = 0.3 + iParam2 * 0.5;  // 0.3 to 0.8
    
    // Calculate vignette factor
    float vignette = smoothstep(radius, radius + 0.4, dist);
    vignette = 1.0 - vignette * intensity;
    
    fragColor = vec4(video.rgb * vignette, video.a);
}
