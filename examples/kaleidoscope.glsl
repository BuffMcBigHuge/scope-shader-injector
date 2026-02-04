// Kaleidoscope - Mirror effect creating kaleidoscope patterns
// iParam1: Number of segments (2 to 16)
// iParam2: Rotation speed
// iParam3: Zoom

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    // Center the coordinates
    vec2 centered = uv - 0.5;
    
    // Convert to polar coordinates
    float angle = atan(centered.y, centered.x);
    float radius = length(centered);
    
    // Number of segments
    float segments = floor(iParam1 * 14.0 + 2.0);  // 2 to 16
    
    // Calculate segment angle
    float segmentAngle = 3.14159 * 2.0 / segments;
    
    // Apply rotation over time
    angle += iTime * iParam2;
    
    // Mirror within segment
    angle = mod(angle, segmentAngle);
    if (angle > segmentAngle * 0.5) {
        angle = segmentAngle - angle;
    }
    
    // Apply zoom
    float zoom = 0.5 + iParam3;
    radius *= zoom;
    
    // Convert back to cartesian
    vec2 newUV = vec2(cos(angle), sin(angle)) * radius + 0.5;
    
    // Sample with wrapping
    newUV = fract(newUV);
    
    vec4 video = texture(iChannel0, newUV);
    fragColor = video;
}
