// Grayscale - Convert video to grayscale
// iParam1: Grayscale intensity (0 = color, 1 = grayscale)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Calculate luminance using perceptual weights
    float luma = dot(video.rgb, vec3(0.299, 0.587, 0.114));
    vec4 gray = vec4(vec3(luma), video.a);
    
    // Mix based on iParam1
    fragColor = mix(video, gray, iParam1);
}
