// Pixelate - Creates a retro pixel art look
// iParam1: Pixel size (larger = more pixelated)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Map param to pixel size (4 to 64 pixels)
    float pixelSize = floor(iParam1 * 60.0 + 4.0);
    
    // Calculate pixelated coordinates
    vec2 pixels = iResolution.xy / pixelSize;
    vec2 uv = floor(fragCoord / pixelSize) * pixelSize / iResolution.xy;
    
    // Sample at pixel center
    uv += (pixelSize * 0.5) / iResolution.xy;
    
    vec4 video = texture(iChannel0, uv);
    fragColor = video;
}
