// Barrel Distortion - Lens distortion effect
// iParam1: Distortion strength (negative for pincushion)
// iParam2: Zoom compensation

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    // Center UV coordinates
    vec2 centered = uv - 0.5;
    
    // Distortion parameters
    float strength = (iParam1 - 0.5) * 2.0;  // -1 to 1
    float zoom = 1.0 + iParam2 * 0.5;         // 1 to 1.5
    
    // Calculate distance from center
    float dist = length(centered);
    
    // Apply barrel/pincushion distortion
    float distortion = 1.0 + strength * dist * dist;
    vec2 distorted = centered * distortion / zoom;
    
    // Convert back to UV space
    vec2 finalUV = distorted + 0.5;
    
    // Check bounds and sample
    if (finalUV.x < 0.0 || finalUV.x > 1.0 || finalUV.y < 0.0 || finalUV.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        fragColor = texture(iChannel0, finalUV);
    }
}
