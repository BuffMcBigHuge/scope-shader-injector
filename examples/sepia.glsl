// Sepia Tone - Classic warm sepia photo effect
// iParam1: Sepia intensity (0 = original, 1 = full sepia)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Sepia tone matrix
    vec3 sepia;
    sepia.r = dot(video.rgb, vec3(0.393, 0.769, 0.189));
    sepia.g = dot(video.rgb, vec3(0.349, 0.686, 0.168));
    sepia.b = dot(video.rgb, vec3(0.272, 0.534, 0.131));
    
    // Mix based on intensity parameter
    fragColor = vec4(mix(video.rgb, sepia, iParam1), video.a);
}
