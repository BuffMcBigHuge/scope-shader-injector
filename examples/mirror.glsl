// Mirror - Split screen with mirroring
// iParam1: Mirror axis (0 = vertical, 1 = horizontal)
// iParam2: Split position (0.5 = center)
// iParam3: Mirror side (0 = left/top mirrors, 1 = right/bottom mirrors)

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    
    float axis = step(0.5, iParam1);  // 0 = vertical split, 1 = horizontal
    float splitPos = iParam2;
    float mirrorSide = step(0.5, iParam3);
    
    vec2 mirroredUV = uv;
    
    if (axis < 0.5) {
        // Vertical split (left/right)
        if (mirrorSide < 0.5) {
            // Mirror left side to right
            if (uv.x > splitPos) {
                mirroredUV.x = splitPos - (uv.x - splitPos);
            }
        } else {
            // Mirror right side to left
            if (uv.x < splitPos) {
                mirroredUV.x = splitPos + (splitPos - uv.x);
            }
        }
    } else {
        // Horizontal split (top/bottom)
        if (mirrorSide < 0.5) {
            // Mirror bottom to top
            if (uv.y > splitPos) {
                mirroredUV.y = splitPos - (uv.y - splitPos);
            }
        } else {
            // Mirror top to bottom
            if (uv.y < splitPos) {
                mirroredUV.y = splitPos + (splitPos - uv.y);
            }
        }
    }
    
    // Clamp to valid range
    mirroredUV = clamp(mirroredUV, 0.0, 1.0);
    
    vec4 video = texture(iChannel0, mirroredUV);
    fragColor = video;
}
