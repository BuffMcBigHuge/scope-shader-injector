// RGB Shift - Animated RGB channel offset
// iParam1: Shift amount
// iParam2: Animation speed
// iParam3: Rotation (0 = horizontal, 1 = circular)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    float amount = iParam1 * 0.03;
    float speed = iParam2 * 3.0;
    float rotation = iParam3;
    
    // Animate offset angle
    float angle = iTime * speed;
    
    // Calculate offset directions
    vec2 rDir, gDir, bDir;
    
    if (rotation < 0.5) {
        // Linear horizontal shift
        rDir = vec2(1.0, 0.0);
        gDir = vec2(0.0, 0.0);
        bDir = vec2(-1.0, 0.0);
    } else {
        // Rotating circular shift
        rDir = vec2(cos(angle), sin(angle));
        gDir = vec2(cos(angle + 2.094), sin(angle + 2.094));  // +120 degrees
        bDir = vec2(cos(angle + 4.189), sin(angle + 4.189));  // +240 degrees
    }
    
    // Sample each channel
    float r = texture(iChannel0, uv + rDir * amount).r;
    float g = texture(iChannel0, uv + gDir * amount).g;
    float b = texture(iChannel0, uv + bDir * amount).b;
    
    fragColor = vec4(r, g, b, 1.0);
}
