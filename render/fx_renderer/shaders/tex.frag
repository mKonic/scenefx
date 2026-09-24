#define SOURCE %d
#define EFFECTS %d

#define SOURCE_TEXTURE_RGBA 1
#define SOURCE_TEXTURE_RGBX 2
#define SOURCE_TEXTURE_EXTERNAL 3

#if !defined(SOURCE) || !defined(EFFECTS)
#error "Missing shader preamble"
#endif

#if SOURCE == SOURCE_TEXTURE_EXTERNAL
#extension GL_OES_EGL_image_external : require
#endif

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_texcoord;

#if SOURCE == SOURCE_TEXTURE_EXTERNAL
uniform samplerExternalOES tex;
#elif SOURCE == SOURCE_TEXTURE_RGBA || SOURCE == SOURCE_TEXTURE_RGBX
uniform sampler2D tex;
#endif

uniform float alpha;

#if EFFECTS
uniform vec2 size;
uniform vec2 position;
uniform float radius_top_left;
uniform float radius_top_right;
uniform float radius_bottom_left;
uniform float radius_bottom_right;

uniform vec2 clip_size;
uniform vec2 clip_position;
uniform float clip_radius_top_left;
uniform float clip_radius_top_right;
uniform float clip_radius_bottom_left;
uniform float clip_radius_bottom_right;
#endif

uniform bool discard_transparent;

// Content that isn't plain SDR (an HDR video, a game's HDR10 swapchain) is
// decoded to linear light, into sRGB primaries, scaled so 1.0 is SDR white,
// and encoded back to gamma 2.2 like everything else, above 1.0 where it's
// brighter. 0: SDR, drawn as is; 1: PQ; 2: linear; 3: gamma 2.2 in
// another gamut.
uniform int hdr_tf;
uniform highp mat3 hdr_prim;
uniform highp float hdr_lum;

highp vec3 pq_decode(highp vec3 e) {
	const highp float m1 = 0.1593017578125;
	const highp float m2 = 78.84375;
	const highp float c1 = 0.8359375;
	const highp float c2 = 18.8515625;
	const highp float c3 = 18.6875;
	highp vec3 p = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
	return pow(max(p - c1, 0.0) / (c2 - c3 * p), vec3(1.0 / m1));
}

// Half precision (these shaders' default) is visibly wrong for PQ.
vec4 hdr_decode(vec4 c) {
	if (hdr_tf == 0) {
		return c;
	}
	highp vec3 rgb = c.a > 0.0 ? c.rgb / c.a : vec3(0.0);
	if (hdr_tf == 1) {
		rgb = pq_decode(rgb);
	} else if (hdr_tf == 3) {
		rgb = pow(rgb, vec3(2.2));
	}
	rgb = hdr_prim * rgb * hdr_lum;
	rgb = sign(rgb) * pow(abs(rgb), vec3(1.0 / 2.2));
	return vec4(rgb * c.a, c.a);
}

vec4 sample_texture() {
#if SOURCE == SOURCE_TEXTURE_RGBA || SOURCE == SOURCE_TEXTURE_EXTERNAL
	return hdr_decode(texture2D(tex, v_texcoord));
#elif SOURCE == SOURCE_TEXTURE_RGBX
	return hdr_decode(vec4(texture2D(tex, v_texcoord).rgb, 1.0));
#endif
}

#if EFFECTS
float corner_alpha(vec2 size, vec2 position, bool is_cutout,
		float radius_tl, float radius_tr, float radius_bl, float radius_br);
#endif

void main() {
#if EFFECTS
	float quad_corner_alpha = corner_alpha(
		size - 0.5,
		position + 0.25,
		false,
		radius_top_left,
		radius_top_right,
		radius_bottom_left,
		radius_bottom_right
	);

	// Clipping
	float clip_corner_alpha = corner_alpha(
		clip_size - 1.0,
		clip_position + 0.5,
		true,
		clip_radius_top_left,
		clip_radius_top_right,
		clip_radius_bottom_left,
		clip_radius_bottom_right
	);

	gl_FragColor = sample_texture() * alpha * quad_corner_alpha * clip_corner_alpha;
#else
	gl_FragColor = sample_texture() * alpha;
#endif

	if (discard_transparent && gl_FragColor.a == 0.0) {
		discard;
	}
}
