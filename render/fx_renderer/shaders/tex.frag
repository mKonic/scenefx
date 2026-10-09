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

// highp: NVIDIA's GLES doesn't define GL_FRAGMENT_PRECISION_HIGH, and its
// mediump is real fp16: texture coordinates and positions past 1024 pixels
// snap to whole or half pixels (see v_frag).
precision highp float;

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
// With discard_transparent: what counts as transparent (Hyprland's
// ignorealpha), 0 for only the fully clear.
uniform float discard_below;

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

vec4 sample_at(vec2 texcoord) {
#if SOURCE == SOURCE_TEXTURE_RGBA || SOURCE == SOURCE_TEXTURE_EXTERNAL
	return texture2D(tex, texcoord);
#elif SOURCE == SOURCE_TEXTURE_RGBX
	return vec4(texture2D(tex, texcoord).rgb, 1.0);
#endif
}

// Motion blur (fx_render_texture_options.motion): what was at each of
// `motion_samples` points back along the way it came (`motion_back`, in
// framebuffer pixels), averaged; the texture fills `motion_box` and is
// clear around it.
uniform int motion_samples;
uniform vec4 motion_box;
uniform vec2 motion_back;
uniform mat3 tex_proj;
#ifndef ATRIUM_V_FRAG
#define ATRIUM_V_FRAG
varying vec2 v_frag;
#endif

// How much of a sample at `p` (pixels into a box `size` big) is inside its
// rounded corners.
float motion_round(vec2 p, vec2 size) {
#if EFFECTS
	float r = 0.0;
	vec2 c = vec2(0.0);
	if (p.x < radius_top_left && p.y < radius_top_left) {
		r = radius_top_left;
		c = vec2(r, r);
	} else if (p.x > size.x - radius_top_right && p.y < radius_top_right) {
		r = radius_top_right;
		c = vec2(size.x - r, r);
	} else if (p.x < radius_bottom_left && p.y > size.y - radius_bottom_left) {
		r = radius_bottom_left;
		c = vec2(r, size.y - r);
	} else if (p.x > size.x - radius_bottom_right && p.y > size.y - radius_bottom_right) {
		r = radius_bottom_right;
		c = vec2(size.x - r, size.y - r);
	}
	if (r > 0.0) {
		return clamp(r - length(p - c) + 0.5, 0.0, 1.0);
	}
#endif
	return 1.0;
}

// A blur's material (fx_render_texture_options.material), Hyprland's frost
// and haze finishes (frostfinish.frag, hazefinish.frag, glassFinish.glsl;
// BSD-3-Clause): 1 frost, 2 haze. The pattern sits at material_box (the
// window); material_texel is a pixel of the texture.
uniform int material;
uniform vec4 material_box;
uniform vec2 material_texel;

float material_hash(vec2 p) {
	vec3 p3 = fract(vec3(p.xyx) * 1689.1984);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

const float FROST_REFRACTION = 20.0;  // decoration:blur:glass:refraction
const float FROST_SIZE = 40.0;        // decoration:blur:glass:size
const float FROST_ROUGHNESS = 1.0;    // decoration:blur:glass:roughness

vec2 frost_random(vec2 cell) {
	return vec2(material_hash(cell + vec2(13.37, 71.91)), material_hash(cell + vec2(83.17, 29.53)));
}

vec2 frost_warp(vec2 p) {
	const vec2 D1 = vec2(1.73, -1.21);
	const vec2 D2 = vec2(1.11, 1.87);
	const vec2 D3 = vec2(-2.19, 0.83);
	vec2 w = vec2(sin(dot(p, D1) + 0.7), cos(dot(p, D2) + 1.9));
	w += vec2(cos(dot(p, D3) + 2.8), sin(dot(p, D1 - D2) + 4.1)) * 0.45;
	return w * 0.16;
}

vec2 frost_gradient(vec2 p) {
	p += frost_warp(p);
	vec2 base = floor(p);
	float nd = 1e10;
	float sd = 1e10;
	vec2 nearest = vec2(0.0);
	vec2 second = vec2(0.0);
	for (int y = -1; y <= 1; ++y) {
		for (int x = -1; x <= 1; ++x) {
			vec2 cell = base + vec2(float(x), float(y));
			vec2 offset = p - (cell + mix(vec2(0.16), vec2(0.84), frost_random(cell)));
			float d = dot(offset, offset);
			if (d < nd) {
				sd = nd;
				second = nearest;
				nd = d;
				nearest = offset;
			} else if (d < sd) {
				sd = d;
				second = offset;
			}
		}
	}
	float boundary = length(second) - length(nearest);
	// fwidth(boundary): about two pixels' worth of cell units.
	float seam = 1.0 - smoothstep(0.0, 0.028 + (2.0 / FROST_SIZE) * 1.5, boundary);
	vec2 dir = second - nearest;
	dir /= max(length(dir), 0.0001);
	vec2 grain = vec2(sin(dot(p, vec2(2.61, -1.43))), cos(dot(p, vec2(1.19, 2.37)))) * 0.09;
	return nearest * 0.34 + grain + dir * seam * 0.55;
}

vec4 frost() {
	vec2 n = frost_gradient((v_frag - material_box.xy) / FROST_SIZE);
	n /= max(1.0, length(n));
	vec4 c = sample_at(clamp(v_texcoord + FROST_REFRACTION * n * material_texel, vec2(0.0), vec2(1.0)));
	const vec2 LIGHT = vec2(-0.451219, 0.892413);
	c.rgb *= 1.0 + dot(n, LIGHT) * FROST_ROUGHNESS * 0.12;
	return c;
}

const float HAZE_INTENSITY = 0.35;    // decoration:blur:haze:intensity
const float HAZE_IRIDESCENCE = 0.7;   // decoration:blur:haze:iridescence
const vec3 BT709_LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 pearl(float phase) {
	const vec3 COOL = vec3(0.55, 1.08, 1.35);
	const vec3 MID = vec3(1.28, 0.86, 1.30);
	const vec3 WARM = vec3(1.40, 0.92, 0.58);
	float t = phase * 0.5 + 0.5;
	vec3 c = t < 0.5 ? mix(COOL, MID, t * 2.0) : mix(MID, WARM, (t - 0.5) * 2.0);
	return c / max(dot(c, BT709_LUMA), 0.001);
}

vec4 haze() {
	vec4 c = sample_at(v_texcoord);
	if (c.a <= 0.001) {
		return c;
	}
	// The frame is gamma 2.2: into light and back.
	vec3 light = pow(max(c.rgb / c.a, vec3(0.0)), vec3(2.2));
	float luminance = max(dot(light, BT709_LUMA), 0.0);
	float phase = clamp(dot(v_texcoord - vec2(0.5), vec2(1.28, -0.72)), -1.0, 1.0);
	vec3 shift = luminance * (pearl(phase) - vec3(1.0)) * HAZE_IRIDESCENCE * 0.65;
	vec3 film = max((light + shift) * (1.0 + (1.0 - abs(phase)) * 0.08), vec3(0.0));
	light = mix(light, film, HAZE_INTENSITY);
	return vec4(pow(max(light, vec3(0.0)), vec3(1.0 / 2.2)) * c.a, c.a);
}

vec4 sample_texture() {
	if (material == 1) {
		return frost();
	} else if (material == 2) {
		return haze();
	}
	if (motion_samples <= 1) {
		return hdr_decode(sample_at(v_texcoord));
	}
	vec4 sum = vec4(0.0);
	for (int k = 0; k < 32; k++) {
		if (k >= motion_samples) {
			break;
		}
		vec2 at = v_frag + motion_back * (float(k) / float(motion_samples - 1));
		vec2 unit = (at - motion_box.xy) / motion_box.zw;
		if (unit.x >= 0.0 && unit.y >= 0.0 && unit.x <= 1.0 && unit.y <= 1.0) {
			sum += sample_at((vec3(unit, 1.0) * tex_proj).xy) * motion_round(unit * motion_box.zw, motion_box.zw);
		}
	}
	return hdr_decode(sum / float(motion_samples));
}

#if EFFECTS
float corner_alpha(vec2 size, vec2 position, bool is_cutout,
		float radius_tl, float radius_tr, float radius_bl, float radius_br);
#endif

// Saturation and brightness (wlr_scene_buffer_set_tint), on premultiplied
// colour, which they scale alike.
uniform vec2 tint;

vec4 tinted(vec4 c) {
	float luma = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
	return vec4(mix(vec3(luma), c.rgb, tint.x) * tint.y, c.a);
}

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

	gl_FragColor = tinted(sample_texture()) * alpha * quad_corner_alpha * clip_corner_alpha;
#else
	gl_FragColor = tinted(sample_texture()) * alpha;
#endif

	if (discard_transparent && gl_FragColor.a <= discard_below) {
		discard;
	}
}
