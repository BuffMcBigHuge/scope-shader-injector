// Fractal Overlay - Mandelbrot fractal blended with video
// iParam1: Fractal zoom
// iParam2: Fractal color intensity
// iParam3: Blend amount

vec3 mandelbrot(vec2 c, float maxIter) {
    vec2 z = vec2(0.0);
    float iter = 0.0;
    
    for (float i = 0.0; i < 100.0; i++) {
        if (i >= maxIter) break;
        
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        
        if (dot(z, z) > 4.0) {
            iter = i;
            break;
        }
        iter = i;
    }
    
    // Smooth coloring
    if (dot(z, z) > 4.0) {
        float smoothIter = iter - log2(log2(dot(z, z))) + 4.0;
        
        // Create color based on iteration count
        float t = smoothIter / maxIter;
        return vec3(
            0.5 + 0.5 * sin(3.0 + t * 6.28 * 2.0),
            0.5 + 0.5 * sin(3.0 + t * 6.28 * 3.0 + 2.0),
            0.5 + 0.5 * sin(3.0 + t * 6.28 * 5.0 + 4.0)
        );
    }
    
    return vec3(0.0);  // Inside the set
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 video = texture(iChannel0, uv);
    
    // Parameters
    float zoom = 0.5 + iParam1 * 3.0;  // 0.5 to 3.5
    float colorIntensity = iParam2;
    float blend = iParam3;
    
    // Map UV to fractal coordinates
    vec2 c = (uv - 0.5) * 3.0 / zoom;
    c.x -= 0.5;  // Center on interesting region
    
    // Animate the view
    c += vec2(sin(iTime * 0.1) * 0.1, cos(iTime * 0.1) * 0.1);
    
    // Calculate fractal
    float maxIter = 50.0 + iParam1 * 50.0;  // More iterations at higher zoom
    vec3 fractal = mandelbrot(c, maxIter) * colorIntensity;
    
    // Blend with video
    vec3 color = mix(video.rgb, video.rgb * 0.5 + fractal, blend);
    
    fragColor = vec4(color, 1.0);
}
