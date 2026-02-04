// Invert Colors - Inverts all RGB values
// iParam1: Inversion strength (0 = original, 1 = fully inverted)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Invert colors
    vec4 inverted = vec4(1.0 - video.rgb, video.a);
    
    // Mix based on iParam1
    fragColor = mix(video, inverted, iParam1);
}
