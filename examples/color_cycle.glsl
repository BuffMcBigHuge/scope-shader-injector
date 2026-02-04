// Color Channel Cycle - Swaps RGB channels over time
// Creates a psychedelic color cycling effect

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Calculate phase based on time
    float phase = iTime * 0.5;
    
    // Create rotation matrix for RGB channels
    float angle = phase * 3.14159 * 2.0;
    mat3 rot = mat3(
        cos(angle), -sin(angle), 0.0,
        sin(angle), cos(angle), 0.0,
        0.0, 0.0, 1.0
    );
    
    // Apply rotation in color space
    vec3 rotated = rot * (video.rgb - 0.5) + 0.5;
    
    // Mix with original based on iParam1
    fragColor = vec4(mix(video.rgb, rotated, iParam1), video.a);
}
