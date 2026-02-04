// Difference - Show difference between video and reference
// iParam1: Difference amplification
// iParam2: Show absolute difference vs signed
// iParam3: Threshold (hide small differences)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    vec4 video = texture(iChannel0, uv);
    vec4 ref = texture(iChannel1, uv);
    
    // Calculate difference
    vec3 diff = video.rgb - ref.rgb;
    
    // Amplification
    float amp = iParam1 * 4.0 + 1.0;  // 1 to 5x
    
    vec3 result;
    if (iParam2 < 0.5) {
        // Absolute difference (black = same, white = different)
        result = abs(diff) * amp;
    } else {
        // Signed difference (gray = same, colors show direction)
        result = diff * 0.5 * amp + 0.5;
    }
    
    // Apply threshold
    float threshold = iParam3 * 0.3;
    float diffMag = length(diff);
    if (diffMag < threshold) {
        result = vec3(0.5);  // Gray for no difference
    }
    
    fragColor = vec4(clamp(result, 0.0, 1.0), 1.0);
}
