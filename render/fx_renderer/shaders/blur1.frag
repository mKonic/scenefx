// highp: NVIDIA's GLES doesn't define GL_FRAGMENT_PRECISION_HIGH, and its
// mediump is real fp16: texture coordinates and positions past 1024 pixels
// snap to whole or half pixels (see v_frag).
precision highp float;

varying highp vec2 v_texcoord;
uniform sampler2D tex;

uniform float radius;
uniform vec2 halfpixel;
// Valid (already rendered) area of the source level, in texture coordinates.
// Samples outside of it would read stale buffer contents.
uniform vec2 uv_min;
uniform vec2 uv_max;

vec4 sample_tex(vec2 p) {
    return texture2D(tex, clamp(p, uv_min, uv_max));
}

void main() {
    vec2 uv = v_texcoord * 2.0;

    vec4 sum = sample_tex(uv) * 4.0;
    sum += sample_tex(uv - halfpixel.xy * radius);
    sum += sample_tex(uv + halfpixel.xy * radius);
    sum += sample_tex(uv + vec2(halfpixel.x, -halfpixel.y) * radius);
    sum += sample_tex(uv - vec2(halfpixel.x, -halfpixel.y) * radius);

    gl_FragColor = sum / 8.0;
}
