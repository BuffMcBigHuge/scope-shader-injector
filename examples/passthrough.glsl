// Passthrough - Simple shader that displays video without modification
// This is the default shader and useful as a template

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Normalized pixel coordinates (from 0 to 1)
    vec2 uv = fragCoord / iResolution.xy;
    
    // Sample video texture (Channel 0)
    vec4 video = texture(iChannel0, uv);
    
    // Output the video as-is
    fragColor = video;
}
