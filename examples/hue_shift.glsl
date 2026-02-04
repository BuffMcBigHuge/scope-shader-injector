// Hue Shift - Rotates colors through the hue spectrum
// iParam1: Manual hue shift (added to time-based shift)
// iParam2: Animation speed (0 = static, 1 = fast)

vec3 rgb2hsv(vec3 c)
{
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Convert to HSV
    vec3 hsv = rgb2hsv(video.rgb);
    
    // Calculate hue shift
    float timeShift = iTime * iParam2 * 0.5;
    float manualShift = iParam1;
    
    // Apply hue rotation
    hsv.x = fract(hsv.x + timeShift + manualShift);
    
    // Convert back to RGB
    vec3 rgb = hsv2rgb(hsv);
    
    fragColor = vec4(rgb, video.a);
}
