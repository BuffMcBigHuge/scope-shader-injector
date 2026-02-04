// Brightness and Contrast adjustment
// iParam1: Brightness (-0.5 to 0.5 mapped from 0-1)
// iParam2: Contrast (0.5 to 2.0 mapped from 0-1)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Map parameters to useful ranges
    float brightness = (iParam1 - 0.5) * 1.0;  // -0.5 to 0.5
    float contrast = iParam2 * 1.5 + 0.5;       // 0.5 to 2.0
    
    // Apply brightness and contrast
    vec3 color = video.rgb;
    color = (color - 0.5) * contrast + 0.5;     // Contrast around midpoint
    color = color + brightness;                  // Add brightness
    
    fragColor = vec4(clamp(color, 0.0, 1.0), video.a);
}
