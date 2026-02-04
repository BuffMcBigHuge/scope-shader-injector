// Dither - Ordered dithering for retro/print look
// iParam1: Dither strength
// iParam2: Pattern size
// iParam3: Color depth (posterization)

// 4x4 Bayer dithering matrix
float bayer4x4(vec2 pos) {
    int x = int(mod(pos.x, 4.0));
    int y = int(mod(pos.y, 4.0));
    
    // Bayer matrix values (0-15 mapped to 0-1)
    float matrix[16] = float[16](
        0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
        12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0,
        3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
        15.0/16.0, 7.0/16.0, 13.0/16.0,  5.0/16.0
    );
    
    return matrix[y * 4 + x];
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Parameters
    float strength = iParam1;
    float patternSize = floor(iParam2 * 7.0 + 1.0);  // 1 to 8
    float levels = floor(iParam3 * 6.0 + 2.0);       // 2 to 8
    
    // Get dither threshold
    vec2 ditherPos = floor(fragCoord / patternSize);
    float threshold = bayer4x4(ditherPos) - 0.5;
    
    // Apply dithering
    vec3 color = video.rgb;
    color += threshold * strength * 0.5;
    
    // Quantize to limited color levels
    color = floor(color * levels) / (levels - 1.0);
    
    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
