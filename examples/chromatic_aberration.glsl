// Chromatic Aberration - RGB channel separation effect
// iParam1: Aberration amount
// iParam2: Radial vs linear mode blend

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec2 centered = uv - 0.5;
    
    // Aberration amount
    float amount = iParam1 * 0.02;
    
    // Direction: radial (from center) vs linear (horizontal)
    vec2 radialDir = normalize(centered) * amount;
    vec2 linearDir = vec2(amount, 0.0);
    
    // Blend between radial and linear based on iParam2
    vec2 dir = mix(linearDir, radialDir * length(centered) * 4.0, iParam2);
    
    // Sample each channel with offset
    float r = texture(iChannel0, uv + dir).r;
    float g = texture(iChannel0, uv).g;
    float b = texture(iChannel0, uv - dir).b;
    
    fragColor = vec4(r, g, b, 1.0);
}
