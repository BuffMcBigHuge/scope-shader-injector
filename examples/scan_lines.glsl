// Scan Lines - CRT/retro monitor effect
// iParam1: Scanline intensity
// iParam2: Scanline thickness
// iParam3: Flicker amount

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Scanline parameters
    float intensity = iParam1;
    float thickness = iParam2 * 4.0 + 1.0;  // 1 to 5 pixels
    float flicker = iParam3 * 0.1;
    
    // Calculate scanline pattern
    float scanline = sin(fragCoord.y * 3.14159 / thickness);
    scanline = scanline * 0.5 + 0.5;  // 0 to 1
    scanline = pow(scanline, 2.0);     // Sharper lines
    
    // Add subtle flicker
    float flickerNoise = sin(iTime * 60.0) * flicker;
    
    // Apply scanlines
    float darkening = 1.0 - intensity * (1.0 - scanline);
    darkening += flickerNoise;
    
    // Slight color fringing on scanlines
    vec3 color = video.rgb * darkening;
    
    // Add subtle green tint for CRT look
    color += vec3(0.0, 0.02, 0.0) * intensity;
    
    fragColor = vec4(color, video.a);
}
