// Posterize - Reduce color levels for a poster/cartoon look
// iParam1: Number of levels (maps to 2-16 levels)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Map param to useful range of levels (2 to 16)
    float levels = floor(iParam1 * 14.0 + 2.0);
    
    // Posterize by quantizing to discrete levels
    vec3 posterized = floor(video.rgb * levels) / (levels - 1.0);
    
    fragColor = vec4(posterized, video.a);
}
