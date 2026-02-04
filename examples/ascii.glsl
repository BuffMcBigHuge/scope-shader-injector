// ASCII Art - Convert video to ASCII-like characters
// iParam1: Character size
// iParam2: Contrast
// iParam3: Color mode (0 = green monochrome, 1 = original colors)

// Pseudo-character based on brightness
float asciiChar(vec2 uv, float brightness) {
    // Quantize brightness to character levels
    // Darker = sparser pattern, brighter = denser pattern
    
    vec2 charUV = fract(uv);
    float level = floor(brightness * 5.0);  // 0-5 levels
    
    // Different patterns for different brightness levels
    float pattern = 0.0;
    
    if (level >= 4.0) {
        // Brightest: nearly solid
        pattern = 1.0;
    } else if (level >= 3.0) {
        // Dense crosshatch
        pattern = step(0.2, max(
            abs(charUV.x - 0.5),
            abs(charUV.y - 0.5)
        )) < 1.0 ? 1.0 : 0.0;
    } else if (level >= 2.0) {
        // Medium: plus sign
        float h = step(0.35, charUV.y) * step(charUV.y, 0.65);
        float v = step(0.35, charUV.x) * step(charUV.x, 0.65);
        pattern = max(h, v);
    } else if (level >= 1.0) {
        // Light: dots
        float dist = length(charUV - 0.5);
        pattern = step(dist, 0.2);
    }
    // level 0: empty (pattern stays 0)
    
    return pattern;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Character grid size
    float charSize = 4.0 + iParam1 * 12.0;  // 4 to 16 pixels
    
    // Calculate character cell
    vec2 charCoord = floor(fragCoord / charSize);
    vec2 charUV = fract(fragCoord / charSize);
    
    // Sample video at cell center
    vec2 sampleUV = (charCoord + 0.5) * charSize / iResolution.xy;
    vec4 video = texture(iChannel0, sampleUV);
    
    // Calculate brightness
    float brightness = dot(video.rgb, vec3(0.299, 0.587, 0.114));
    
    // Apply contrast
    float contrast = iParam2 * 2.0 + 0.5;  // 0.5 to 2.5
    brightness = (brightness - 0.5) * contrast + 0.5;
    brightness = clamp(brightness, 0.0, 1.0);
    
    // Get character pattern
    float char = asciiChar(charUV, brightness);
    
    // Color mode
    vec3 color;
    if (iParam3 < 0.5) {
        // Green monochrome terminal look
        color = vec3(0.0, char * brightness, 0.0);
        // Add subtle glow
        color += vec3(0.0, 0.1, 0.05) * char;
    } else {
        // Colored ASCII
        color = video.rgb * char;
    }
    
    // Dark background
    color = max(color, vec3(0.02));
    
    fragColor = vec4(color, 1.0);
}
