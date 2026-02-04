// Glitch Effect - Digital glitch distortion
// iParam1: Glitch intensity
// iParam2: Glitch frequency (how often glitches occur)
// iParam3: Color separation amount

// Simple hash function for pseudo-random numbers
float hash(float n) {
    return fract(sin(n) * 43758.5453);
}

float hash2(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    float intensity = iParam1;
    float frequency = iParam2;
    float colorSep = iParam3 * 0.05;
    
    // Time-based glitch trigger
    float glitchTime = floor(iTime * (5.0 + frequency * 20.0));
    float glitchRand = hash(glitchTime);
    
    // Only glitch some of the time
    bool isGlitching = glitchRand > (1.0 - frequency);
    
    vec2 glitchedUV = uv;
    
    if (isGlitching && intensity > 0.0) {
        // Block-based offset
        float blockY = floor(uv.y * 20.0);
        float blockRand = hash(blockY + glitchTime);
        
        if (blockRand > 0.5) {
            // Horizontal shift
            float shift = (hash(blockY * 2.0 + glitchTime) - 0.5) * intensity * 0.2;
            glitchedUV.x += shift;
        }
        
        // Random horizontal line displacement
        if (hash(floor(fragCoord.y) + glitchTime * 100.0) > 0.97) {
            glitchedUV.x += (hash(fragCoord.y) - 0.5) * intensity * 0.3;
        }
    }
    
    // Sample with color separation
    float r = texture(iChannel0, glitchedUV + vec2(colorSep, 0.0)).r;
    float g = texture(iChannel0, glitchedUV).g;
    float b = texture(iChannel0, glitchedUV - vec2(colorSep, 0.0)).b;
    
    vec3 color = vec3(r, g, b);
    
    // Add occasional color noise
    if (isGlitching && hash(fragCoord.x + glitchTime) > 0.99) {
        color = vec3(hash(fragCoord.x + fragCoord.y + iTime));
    }
    
    fragColor = vec4(color, 1.0);
}
