// Mask with Reference - Use reference image brightness as mask
// iParam1: Mask threshold
// iParam2: Mask softness
// iParam3: Invert mask

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    vec4 video = texture(iChannel0, uv);
    vec4 ref = texture(iChannel1, uv);
    
    // Calculate mask from reference luminance
    float mask = dot(ref.rgb, vec3(0.299, 0.587, 0.114));
    
    // Apply threshold and softness
    float threshold = iParam1;
    float softness = iParam2 * 0.5 + 0.01;  // Small minimum to avoid division issues
    
    mask = smoothstep(threshold - softness, threshold + softness, mask);
    
    // Optionally invert mask
    if (iParam3 > 0.5) {
        mask = 1.0 - mask;
    }
    
    // Apply mask - show video where mask is bright
    fragColor = vec4(video.rgb * mask, 1.0);
}
