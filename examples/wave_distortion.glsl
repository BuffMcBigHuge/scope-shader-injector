// Wave Distortion - Animated wavy distortion effect
// iParam1: Wave amplitude
// iParam2: Wave frequency
// iParam3: Animation speed

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    // Parameters
    float amplitude = iParam1 * 0.1;           // 0 to 0.1
    float frequency = iParam2 * 20.0 + 5.0;    // 5 to 25
    float speed = iParam3 * 5.0;               // 0 to 5
    
    // Calculate wave offset
    float waveX = sin(uv.y * frequency + iTime * speed) * amplitude;
    float waveY = cos(uv.x * frequency + iTime * speed * 0.7) * amplitude;
    
    // Apply distortion
    vec2 distortedUV = uv + vec2(waveX, waveY);
    
    // Clamp UV to valid range
    distortedUV = clamp(distortedUV, 0.0, 1.0);
    
    vec4 video = texture(iChannel0, distortedUV);
    fragColor = video;
}
