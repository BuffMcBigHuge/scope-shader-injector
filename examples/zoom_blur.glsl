// Zoom Blur - Radial blur emanating from center
// iParam1: Blur strength
// iParam2: Center X (0.5 = center)
// iParam3: Center Y (0.5 = center)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    // Blur center
    vec2 center = vec2(iParam2, iParam3);
    
    // Direction from center
    vec2 dir = uv - center;
    
    // Blur strength
    float strength = iParam1 * 0.1;
    
    // Number of samples
    const int samples = 16;
    
    vec4 color = vec4(0.0);
    float totalWeight = 0.0;
    
    for (int i = 0; i < samples; i++) {
        float t = float(i) / float(samples - 1);
        float weight = 1.0 - t;  // Closer samples weighted more
        
        vec2 offset = dir * strength * t;
        color += texture(iChannel0, uv - offset) * weight;
        totalWeight += weight;
    }
    
    fragColor = color / totalWeight;
}
