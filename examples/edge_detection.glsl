// Edge Detection - Sobel edge detection filter
// iParam1: Edge intensity
// iParam2: Mix with original (0 = edges only, 1 = edges + original)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec2 texel = 1.0 / iResolution.xy;
    
    // Sample 3x3 neighborhood
    float tl = dot(texture(iChannel0, uv + texel * vec2(-1, 1)).rgb, vec3(0.299, 0.587, 0.114));
    float t  = dot(texture(iChannel0, uv + texel * vec2(0, 1)).rgb, vec3(0.299, 0.587, 0.114));
    float tr = dot(texture(iChannel0, uv + texel * vec2(1, 1)).rgb, vec3(0.299, 0.587, 0.114));
    float l  = dot(texture(iChannel0, uv + texel * vec2(-1, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float r  = dot(texture(iChannel0, uv + texel * vec2(1, 0)).rgb, vec3(0.299, 0.587, 0.114));
    float bl = dot(texture(iChannel0, uv + texel * vec2(-1, -1)).rgb, vec3(0.299, 0.587, 0.114));
    float b  = dot(texture(iChannel0, uv + texel * vec2(0, -1)).rgb, vec3(0.299, 0.587, 0.114));
    float br = dot(texture(iChannel0, uv + texel * vec2(1, -1)).rgb, vec3(0.299, 0.587, 0.114));
    
    // Sobel operators
    float gx = -tl - 2.0*l - bl + tr + 2.0*r + br;
    float gy = -tl - 2.0*t - tr + bl + 2.0*b + br;
    
    // Edge magnitude
    float edge = sqrt(gx*gx + gy*gy) * (iParam1 * 4.0 + 1.0);
    
    // Original video
    vec4 video = texture(iChannel0, uv);
    
    // Mix edges with original
    vec3 edgeColor = vec3(edge);
    vec3 result = mix(edgeColor, video.rgb + edgeColor * 0.5, iParam2);
    
    fragColor = vec4(result, 1.0);
}
