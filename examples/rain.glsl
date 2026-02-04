// Rain Effect - Animated rain/water droplet overlay
// iParam1: Rain density
// iParam2: Rain speed
// iParam3: Droplet size

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    float density = iParam1 * 100.0 + 10.0;  // 10 to 110 drops
    float speed = iParam2 * 2.0 + 0.5;        // 0.5 to 2.5
    float dropSize = iParam3 * 0.02 + 0.005;  // 0.005 to 0.025
    
    // Animated UV for rain motion
    float time = iTime * speed;
    
    // Create multiple rain layers for depth
    float rainEffect = 0.0;
    vec2 distortion = vec2(0.0);
    
    for (int layer = 0; layer < 3; layer++) {
        float layerScale = 1.0 + float(layer) * 0.5;
        float layerSpeed = 1.0 + float(layer) * 0.3;
        
        // Grid for rain drops
        vec2 rainUV = uv * vec2(density * layerScale, density * 0.5);
        rainUV.y += time * layerSpeed * 2.0;
        
        vec2 gridId = floor(rainUV);
        vec2 gridUV = fract(rainUV) - 0.5;
        
        // Randomize drop position within cell
        float rand = hash(gridId);
        vec2 dropOffset = vec2(rand - 0.5, hash(gridId + 100.0) - 0.5) * 0.8;
        
        // Only show some drops
        if (rand > 0.3) {
            vec2 dropPos = gridUV - dropOffset;
            float drop = smoothstep(dropSize * layerScale, 0.0, length(dropPos));
            
            rainEffect += drop * (1.0 - float(layer) * 0.2);
            
            // Add refraction distortion
            if (drop > 0.0) {
                distortion += dropPos * drop * 0.02;
            }
        }
    }
    
    // Apply distortion to video
    vec2 distortedUV = uv + distortion;
    vec4 distortedVideo = texture(iChannel0, clamp(distortedUV, 0.0, 1.0));
    
    // Blend rain effect
    vec3 color = distortedVideo.rgb;
    color += rainEffect * 0.3;  // Add rain highlight
    
    fragColor = vec4(color, 1.0);
}
