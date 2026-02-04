// Matrix - Digital rain / Matrix-style effect
// iParam1: Character density
// iParam2: Rain speed
// iParam3: Blend with video

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

// Simple character pattern (fake glyphs)
float character(vec2 uv, float seed) {
    // Create a pseudo-character using noise patterns
    float char = 0.0;
    
    // Vertical bars
    float bars = step(0.7, fract(uv.x * 3.0 + seed));
    // Horizontal segments  
    float segs = step(0.5, fract(uv.y * 4.0 + seed * 2.0));
    
    char = bars * segs + (1.0 - bars) * (1.0 - segs) * step(0.6, seed);
    
    return char * step(0.1, uv.x) * step(uv.x, 0.9) * step(0.1, uv.y) * step(uv.y, 0.9);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Parameters
    float density = 20.0 + iParam1 * 40.0;  // 20 to 60 columns
    float speed = 2.0 + iParam2 * 8.0;       // 2 to 10
    float blend = iParam3;
    
    // Create character grid
    vec2 charSize = vec2(density, density * 2.0);  // Characters are taller than wide
    vec2 gridPos = uv * charSize;
    vec2 cellId = floor(gridPos);
    vec2 cellUV = fract(gridPos);
    
    // Animate each column independently
    float colSeed = hash(vec2(cellId.x, 0.0));
    float rowOffset = iTime * speed * (0.5 + colSeed * 0.5);
    
    // Calculate which character is shown
    float charId = floor(cellId.y + rowOffset);
    float charSeed = hash(vec2(cellId.x, charId));
    
    // Character brightness (fades as it falls)
    float brightness = fract(cellId.y / 20.0 + rowOffset / 20.0);
    brightness = pow(brightness, 0.5);  // Adjust falloff
    
    // Leading character is brighter
    float isLeading = step(0.95, fract(rowOffset / 20.0 - cellId.y / 20.0));
    brightness = max(brightness, isLeading);
    
    // Only show characters in some columns
    float showColumn = step(0.3, colSeed);
    
    // Generate character pattern
    float char = character(cellUV, charSeed);
    
    // Matrix green color
    vec3 matrixColor = vec3(0.0, 1.0, 0.3) * brightness * char * showColumn;
    
    // Add glow
    matrixColor += vec3(0.0, 0.5, 0.2) * brightness * showColumn * 0.3;
    
    // Blend with video
    vec3 color = mix(matrixColor, video.rgb * 0.5 + matrixColor, blend);
    
    fragColor = vec4(color, 1.0);
}
