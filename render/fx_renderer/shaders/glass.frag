// Liquid Glass as Apple's UI kit builds it (the iOS 27 kit's liquid-glass
// styles): not a lens over a bare backdrop but a layered material. What is
// behind is frosted, made richer and brighter, and bent inward near the
// edge; fills are laid over it in their blend modes (a floor of colour, a
// luminosity wash), then inner shadows give the slab its thickness (soft
// glows along its top and bottom, hairline highlights) and outer ones set it
// off what is behind (hairlines, a deep soft drop shadow). The look holds
// over a plain backdrop, where a lens alone has nothing to bend.
//
// A port of atrium-os's renderd glass.frag. renderd draws each glass layer
// over the finished scene below it; here a node's shapes are drawn in one
// pass, bottom to top, each over what the ones before it made. A shape on
// earlier glass takes that glass as its backdrop (frosted already, so near
// enough what blurring it again would give).
//
// Written opaque over the node's region: `sharp` is a copy of what was there.

// highp: NVIDIA's GLES doesn't define GL_FRAGMENT_PRECISION_HIGH, and its
// mediump is real fp16: texture coordinates and positions past 1024 pixels
// snap to whole or half pixels (see v_frag).
precision highp float;

varying vec2 v_texcoord;
#ifndef ATRIUM_V_FRAG
#define ATRIUM_V_FRAG
// Where this fragment is, in framebuffer pixels: gl_FragCoord, but highp
// (GLSL ES 1.00's is mediump, which NVIDIA honours: past 1024 pixels it
// rounds pixel centres in pairs, so every curve there was drawn at half
// resolution).
varying vec2 v_frag;
#endif

uniform sampler2D tex;    // the backdrop, frosted: the whole buffer
uniform sampler2D sharp;  // the backdrop as it is
uniform vec2 texel;       // one buffer pixel, in tex coordinates
uniform float alpha;      // the node's: glass fading in or out

// The shapes, bottom to top: x, y, width, height; radius, opacity, material;
// where each shows (x, y, width, height; width < 0: all of it), cut there.
uniform vec4 shape_box[16];
uniform vec4 shape_extra[16];
uniform vec4 shape_clip[16];
uniform int shape_count;

// The materials (WLR_SCENE_GLASS_MAX_MATERIALS), in buffer pixels:
// saturation, brightness, depth, distortion; fill count, shadow count, how
// far the outer shadows reach. Fills 4 a material, shadows 16.
uniform vec4 mat_optics[5];
uniform vec4 mat_counts[5];
uniform vec4 fill_color[20];  // straight rgb, alpha: how much
uniform float fill_blend[20];
uniform vec4 sh_geo[80];      // offset x, y, sigma, spread
uniform vec4 sh_color[80];    // straight rgb, alpha: how much
uniform vec2 sh_meta[80];     // blend, inner

float sd_round_rect(vec2 p, vec2 half_size, float r) {
	r = min(r, min(half_size.x, half_size.y));
	vec2 q = abs(p) - half_size + r;
	return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

float erf_approx(float x) {
	float s = sign(x), a = abs(x);
	x = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
	x *= x;
	return s - s / (x * x);
}

// Coverage of the rounded rect [lower, upper] blurred by `sigma`, at p: the
// Gaussian's CDF across its edge (exact for a straight edge).
float box_shadow(vec2 lower, vec2 upper, vec2 p, float sigma, float corner) {
	vec2 hs = max((upper - lower) * 0.5, vec2(0.0));
	float d = sd_round_rect(p - (lower + upper) * 0.5, hs, corner);
	if (sigma < 0.25) {
		return clamp(0.5 - d, 0.0, 1.0);
	}
	return 0.5 - 0.5 * erf_approx(d * 0.70710678 / sigma);
}

// Blend modes (W3C Compositing Level 1 + Apple's plus-darker/plus-lighter),
// numbered as Sketch's: b is the backdrop, s the source, both straight.
float lum(vec3 c) {
	return dot(c, vec3(0.3, 0.59, 0.11));
}
vec3 clip_color(vec3 c) {
	float l = lum(c), n = min(min(c.r, c.g), c.b), x = max(max(c.r, c.g), c.b);
	if (n < 0.0) c = l + (c - l) * l / max(l - n, 1e-5);
	if (x > 1.0) c = l + (c - l) * (1.0 - l) / max(x - l, 1e-5);
	return c;
}
vec3 set_lum(vec3 c, float l) {
	return clip_color(c + (l - lum(c)));
}
float sat(vec3 c) {
	return max(max(c.r, c.g), c.b) - min(min(c.r, c.g), c.b);
}
vec3 set_sat(vec3 c, float s) {
	float mx = max(max(c.r, c.g), c.b), mn = min(min(c.r, c.g), c.b);
	return mx > mn ? (c - mn) * s / (mx - mn) : vec3(0.0);
}
float soft_light1(float b, float s) {
	if (s <= 0.5) return b - (1.0 - 2.0 * s) * b * (1.0 - b);
	float d = b <= 0.25 ? ((16.0 * b - 12.0) * b + 4.0) * b : sqrt(b);
	return b + (2.0 * s - 1.0) * (d - b);
}
vec3 blend_rgb(int mode, vec3 b, vec3 s) {
	if (mode == 1) return min(b, s);                                       // darken
	if (mode == 2) return b * s;                                           // multiply
	if (mode == 3) return 1.0 - min(vec3(1.0), (1.0 - b) / max(s, 1e-5));  // color burn
	if (mode == 4) return max(b, s);                                       // lighten
	if (mode == 5) return b + s - b * s;                                   // screen
	if (mode == 6) return min(vec3(1.0), b / max(1.0 - s, 1e-5));          // color dodge
	if (mode == 7) return mix(2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s), step(0.5, b));  // overlay
	if (mode == 8) return vec3(soft_light1(b.r, s.r), soft_light1(b.g, s.g), soft_light1(b.b, s.b));
	if (mode == 9) return mix(2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s), step(0.5, s));  // hard light
	if (mode == 10) return abs(b - s);                                     // difference
	if (mode == 11) return b + s - 2.0 * b * s;                            // exclusion
	if (mode == 12) return set_lum(set_sat(s, sat(b)), lum(b));            // hue
	if (mode == 13) return set_lum(set_sat(b, sat(s)), lum(b));            // saturation
	if (mode == 14) return set_lum(s, lum(b));                             // color
	if (mode == 15) return set_lum(b, lum(s));                             // luminosity
	if (mode == 16) return max(vec3(0.0), b + s - 1.0);                    // plus darker
	if (mode == 17) return min(vec3(1.0), b + s);                          // plus lighter
	return s;                                                              // normal
}
// Source s with coverage a composited onto the opaque backdrop b.
vec3 blend(int mode, vec3 b, vec3 s, float a) {
	return mix(b, blend_rgb(mode, b, s), a);
}

// A buffer's colour at uv, straight (its edges may be transparent).
vec3 sample_rgb(sampler2D t, vec2 uv) {
	vec4 c = texture2D(t, uv);
	return c.a > 0.0 ? c.rgb / c.a : vec3(0.0);
}

void main() {
	vec2 p = v_frag;
	vec3 backdrop = sample_rgb(sharp, v_texcoord);
	vec3 c = backdrop;
	float under = 0.0;  // how much of c is glass already
	for (int i = 0; i < 16; i++) {
		if (i >= shape_count) {
			break;
		}
		vec4 cl = shape_clip[i];
		if (cl.z >= 0.0 && (p.x < cl.x || p.y < cl.y || p.x >= cl.x + cl.z || p.y >= cl.y + cl.w)) {
			continue;
		}
		vec4 b = shape_box[i];
		vec4 e = shape_extra[i];
		int m = int(e.z + 0.5);
		vec2 half_size = 0.5 * b.zw;
		vec2 rel = p - (b.xy + half_size);
		float radius = min(e.x, min(half_size.x, half_size.y));
		float d = sd_round_rect(rel, half_size, radius);
		vec4 counts = mat_counts[m];
		if (d > counts.z) {
			continue;  // past its outer shadows
		}
		int nf = int(counts.x + 0.5), ns = int(counts.y + 0.5);

		// Outside: the outer shadows, over what is behind.
		vec3 outside = c;
		for (int k = 0; k < 16; k++) {
			if (k >= ns) {
				break;
			}
			int j = m * 16 + k;
			if (sh_meta[j].y > 0.5) {
				continue;
			}
			vec4 geo = sh_geo[j];
			vec2 lo = b.xy + geo.xy - geo.w, hi = b.xy + b.zw + geo.xy + geo.w;
			float a = box_shadow(lo, hi, p, geo.z, max(radius + geo.w, 0.0)) * sh_color[j].a;
			outside = blend(int(sh_meta[j].x + 0.5), outside, sh_color[j].rgb, a);
		}

		vec3 result = outside;
		float inside = clamp(0.5 - d, 0.0, 1.0);
		if (inside > 0.0) {
			// Bent: within `depth` of the edge it shows what lies further
			// in, most at the rim, nothing at the band's inner end.
			vec4 o = mat_optics[m];
			float band = max(o.z * min(half_size.x, half_size.y), 1.0);
			float t = clamp(-d / band, 0.0, 1.0);
			vec2 q = abs(rel) - half_size + radius;
			vec2 n = (q.x > 0.0 || q.y > 0.0) ? normalize(max(q, 0.0))
				: (q.x > q.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
			n *= vec2(rel.x < 0.0 ? -1.0 : 1.0, rel.y < 0.0 ? -1.0 : 1.0);
			vec2 off = -n * o.w * band * (1.0 - t) * (1.0 - t);
			vec3 glass = mix(sample_rgb(tex, v_texcoord + off * texel), c, under);
			glass = mix(vec3(lum(glass)), glass, o.x) * o.y;
			for (int k = 0; k < 4; k++) {
				if (k >= nf) {
					break;
				}
				int j = m * 4 + k;
				glass = blend(int(fill_blend[j] + 0.5), glass, fill_color[j].rgb, fill_color[j].a);
			}
			// Inner shadows: what lies outside the (offset, grown or
			// shrunk) shape, blurred, cast inward.
			for (int k = 0; k < 16; k++) {
				if (k >= ns) {
					break;
				}
				int j = m * 16 + k;
				if (sh_meta[j].y < 0.5) {
					continue;
				}
				vec4 geo = sh_geo[j];
				vec2 lo = b.xy + geo.xy + geo.w, hi = b.xy + b.zw + geo.xy - geo.w;
				float hole = box_shadow(lo, hi, p, geo.z, max(radius - geo.w, 0.0));
				glass = blend(int(sh_meta[j].x + 0.5), glass, sh_color[j].rgb, (1.0 - hole) * sh_color[j].a);
			}
			result = mix(outside, glass, inside);
		}
		c = mix(c, result, e.y);
		under = max(under, inside * e.y);
	}
	gl_FragColor = vec4(mix(backdrop, clamp(c, 0.0, 1.0), alpha), 1.0);
}
