// Night Vision - Military night vision goggle effect
// iParam1: Brightness amplification
// iParam2: Noise intensity
// iParam3: Vignette strength

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Parameters
    float amplification = 1.0 + iParam1 * 4.0;  // 1 to 5x
    float noiseAmount = iParam2 * 0.3;
    float vignetteStrength = iParam3;
    
    // Convert to luminance
    float luma = dot(video.rgb, vec3(0.299, 0.587, 0.114));
    
    // Amplify brightness
    luma = pow(luma, 0.7) * amplification;
    
    // Add noise
    float noise = hash(fragCoord + vec2(iTime * 100.0, 0.0)) * 2.0 - 1.0;
    luma += noise * noiseAmount * (0.5 + luma * 0.5);
    
    // Add scanline effect
    float scanline = sin(fragCoord.y * 2.0) * 0.03;
    luma += scanline;
    
    // Night vision green tint
    vec3 color = vec3(luma * 0.2, luma * 1.0, luma * 0.2);
    
    // Circular vignette (goggles effect)
    vec2 centered = uv - 0.5;
    float dist = length(centered);
    float vignette = 1.0 - smoothstep(0.3, 0.5, dist) * vignetteStrength;
    
    // Hard edge for goggle look
    if (dist > 0.48) {
        vignette = 0.0;
    }
    
    color *= vignette;
    
    // Add subtle phosphor glow
    color += vec3(0.0, 0.02, 0.0);
    
    fragColor = vec4(color, 1.0);
}
