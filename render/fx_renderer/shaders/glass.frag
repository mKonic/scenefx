// Liquid Glass, after Apple's (WWDC25 "Meet Liquid Glass"): a lens, not a
// frosted pane. The panel is a slab of glass (n = 1.5) whose rim is a convex
// squircle bevel, y = (1 - (1 - x)^4)^(1/4): flat inside, where what is
// behind shows through untouched, curving down at the edge, where light is
// refracted (Snell's law) so the rim shows what lies further in, compressed.
// A thin line along the rim catches the light, coloured by what is behind
// (Apple's specular is the backdrop, far more saturated, not white); a soft
// shadow falls outside. A tint that adapts to what is behind keeps what is
// drawn on the glass legible.
//
// The shape is the panel's own buffer: any pixel it draws is glass (the
// shell draws no shadow of its own under glass). `field` is that shape
// blurred by a Gaussian of `field_sigma` (glass_field.frag): along a straight
// edge it is the normal CDF of the distance to it, which gives a smooth
// distance, the edge's direction (its gradient) and the shadow.

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

uniform sampler2D tex;   // the (lightly) blurred background, the whole buffer
uniform sampler2D mask;  // the panel's buffer
uniform bool has_mask;
uniform sampler2D field; // the shape, blurred
uniform vec2 field_texel;
uniform float field_sigma;
uniform vec2 texel;      // one buffer pixel, in tex coordinates
uniform vec2 box_pos;    // the panel, in buffer pixels
uniform vec2 box_size;
uniform vec4 mask_src;   // the mask's source box, normalized
uniform float radius;    // corners, without a mask
uniform float refraction; // the slab's height, in pixels: how far light bends
uniform float thickness;  // the bevel's width, in pixels
uniform float alpha;

uniform vec4 tint;        // the glass's own colour (straight alpha: how much)
uniform float adapt;      // how much more tint where the backdrop fights it
uniform float saturation; // vibrancy: the backdrop's colour, a little richer
uniform float highlight;  // the rim's light
uniform vec2 light_dir;   // where the light travels, screen space (y down)
uniform float shadow;     // the shadow's darkness under the glass

// The glass's exact shapes (atrium-glass-v1), when given: x, y, width,
// height; radius, opacity. Otherwise the shape is the mask, through `field`.
uniform vec4 shape_box[16];
uniform vec2 shape_extra[16];
// Where each shows (x, y, width, height; width < 0: all of it): cut there,
// with no rim along the cut, as a card scrolled half out of its list.
uniform vec4 shape_clip[16];
uniform int shape_count;

// 1 on glass, 0 off it, antialiased across the edge's pixel.
float cover(vec2 p) {
	if (has_mask) {
		vec2 r = (p - box_pos) / box_size;
		if (r.x < 0.0 || r.y < 0.0 || r.x > 1.0 || r.y > 1.0) {
			return 0.0;
		}
		// Linear in the panel's alpha up to its glass fill's (atrium's shell
		// fills glass at 6%), so the edge's antialiasing carries through.
		return clamp(texture2D(mask, mask_src.xy + r * mask_src.zw).a * 18.0, 0.0, 1.0);
	}
	vec2 q = abs(p - box_pos - box_size * 0.5) - box_size * 0.5 + radius;
	float d = min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
	return clamp(0.5 - d, 0.0, 1.0);
}

float shape(vec2 p) {
	return texture2D(field, p * field_texel).r;
}

// The inverse error function (Winitzki's approximation, to about 0.2%).
float erfinv(float x) {
	float l = log(max(1.0 - x * x, 1e-7));
	float t = 2.0 / (3.14159265 * 0.147) + 0.5 * l;
	return sign(x) * sqrt(max(sqrt(t * t - l / 0.147) - t, 0.0));
}

// How far inside the edge the field's value puts p (negative outside).
float depth_in(float f) {
	f = clamp(f, 0.0005, 0.9995);
	return field_sigma * 1.41421356 * erfinv(2.0 * f - 1.0);
}

float squircle(float x) {
	return pow(1.0 - pow(1.0 - clamp(x, 0.0, 1.0), 4.0), 0.25);
}

// How far a ray straight down through the bevel, at x (0 at the rim, 1
// where the glass turns flat), lands from where it entered.
float bend(float x, float bevel) {
	float e = 0.002;
	float x0 = clamp(x, e, 1.0 - e);
	float slope = (squircle(x0 + e) - squircle(x0 - e)) / (2.0 * e);
	// The surface's tilt: its slope, stretched over the bevel's width.
	float tilt = atan(slope * refraction / bevel);
	float inside = asin(sin(tilt) / 1.5);
	// Through the bevel's height here, and the slab's base under it.
	float depth = refraction * (0.5 + 0.5 * squircle(x0));
	return depth * tan(tilt - inside);
}

// The shape this pixel belongs to: the last (topmost, as the shell paints
// them) that reaches it, else the nearest, for its shadow. Its signed
// distance (negative inside, exact for rounded rectangles), outward normal,
// opacity and bevel width.
float shapes_sd(vec2 p, out vec2 normal, out float opacity, out float bevel) {
	float best = 1e5;
	float chosen = 1e5;
	bool inside_any = false;
	normal = vec2(0.0);
	opacity = 0.0;
	bevel = 1.0;
	for (int i = 0; i < 16; i++) {
		if (i >= shape_count) {
			break;
		}
		vec4 c = shape_clip[i];
		if (c.z >= 0.0 && (p.x < c.x || p.y < c.y || p.x >= c.x + c.z || p.y >= c.y + c.w)) {
			continue;
		}
		vec4 b = shape_box[i];
		vec2 half_size = 0.5 * b.zw;
		float r = min(shape_extra[i].x, min(half_size.x, half_size.y));
		vec2 rel = p - (b.xy + half_size);
		vec2 q = abs(rel) - half_size + r;
		float sd = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
		bool inside = sd < 0.5;
		if (inside || (!inside_any && sd < best)) {
			best = min(best, sd);
			inside_any = inside_any || inside;
			chosen = sd;
			vec2 n = (q.x > 0.0 || q.y > 0.0) ? normalize(max(q, 0.0))
				: (q.x > q.y ? vec2(1.0, 0.0) : vec2(0.0, 1.0));
			normal = n * vec2(rel.x < 0.0 ? -1.0 : 1.0, rel.y < 0.0 ? -1.0 : 1.0);
			opacity = shape_extra[i].y;
			// A bevel no wider than the corner's curve, so it stays smooth.
			bevel = max(min(thickness, r), 1.0);
		}
	}
	return chosen;
}

void main() {
	vec2 p = v_frag;
	float here;
	float shade;
	float d;
	float bevel = max(thickness, 1.0);
	vec2 outward;
	// How much of what's covered is the rim's line, from area (a line a
	// pixel or two wide must be, or it flickers along curves).
	float rim_share;
	const float RIM = 1.3;
	if (shape_count > 0) {
		float opacity;
		float sd = shapes_sd(p, outward, opacity, bevel);
		// Exact coverage of this pixel, and a soft shadow falling off
		// over the bevel's width and more.
		here = clamp(0.5 - sd, 0.0, 1.0) * opacity;
		float reach = 1.2 * max(thickness, 1.0);
		shade = shadow * opacity * pow(1.0 - clamp(sd / reach, 0.0, 1.0), 2.0);
		d = max(-sd, 0.0);
		float full = clamp(0.5 - sd, 0.0, 1.0);
		float band = full - clamp(0.5 - (sd + RIM), 0.0, 1.0);
		rim_share = full > 0.0 ? band / full : 0.0;
	} else {
		here = cover(p);
		float f = shape(p);
		// Off the glass: its shadow, the blurred shape itself.
		shade = shadow * pow(clamp(2.0 * f, 0.0, 1.0), 1.5);
		// The way out, across the field's slope.
		vec2 grad = vec2(shape(p + vec2(1.0, 0.0)) - shape(p - vec2(1.0, 0.0)),
			shape(p + vec2(0.0, 1.0)) - shape(p - vec2(0.0, 1.0)));
		outward = length(grad) > 1e-5 ? -normalize(grad) : vec2(0.0);
		d = max(depth_in(f), 0.0);
		rim_share = 1.0 - smoothstep(0.3, 1.8, d);
	}
	if (here <= 0.0) {
		gl_FragColor = vec4(0.0, 0.0, 0.0, shade) * alpha;
		return;
	}
	float x = d / bevel;

	// The lens: what lies further in shows here, the more the bevel tilts.
	// Near the rim it squeezes several pixels of the backdrop into one, so
	// average over all of them (the pixel's footprint, from its outer edge
	// to its inner one, mapped through the lens) rather than pick one:
	// otherwise the rim sparkles and steps.
	vec3 rgb = vec3(0.0);
	float coverage_a = 0.0;
	if (x < 1.0) {
		float scale = bevel / max(thickness, 1.0);
		float d0 = max(d - 0.5, 0.0);
		float d1 = d + 0.5;
		float u0 = d0 + bend(d0 / bevel, bevel) * scale;
		float u1 = d1 + bend(d1 / bevel, bevel) * scale;
		for (int k = 0; k < 6; k++) {
			float u = mix(u0, u1, (float(k) + 0.5) / 6.0);
			vec2 off = -outward * (u - d);
			vec4 c = texture2D(tex, v_texcoord + off * texel);
			// Each colour bent a little apart.
			c.r = texture2D(tex, v_texcoord + off * 1.04 * texel).r;
			c.b = texture2D(tex, v_texcoord + off * 0.96 * texel).b;
			rgb += c.rgb;
			coverage_a += c.a;
		}
		rgb /= 6.0;
		coverage_a /= 6.0;
	} else {
		vec4 c = texture2D(tex, v_texcoord);
		rgb = c.rgb;
		coverage_a = c.a;
	}
	rgb = coverage_a > 0.0 ? rgb / coverage_a : vec3(0.0);

	// Vibrancy.
	float y = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
	rgb = clamp(mix(vec3(y), rgb, saturation), 0.0, 1.0);
	vec3 backdrop = rgb;

	// The adaptive tint: more where the backdrop is close to what's drawn on
	// the glass (bright behind dark glass, dark behind light).
	float tint_y = dot(tint.rgb, vec3(0.2126, 0.7152, 0.0722));
	float fight = tint_y < 0.5 ? smoothstep(0.3, 0.9, y) : smoothstep(0.7, 0.1, y);
	rgb = mix(rgb, tint.rgb, clamp(tint.a + adapt * fight, 0.0, 0.92));

	// The rim: a line a pixel or two wide that catches the light where it
	// faces it and, fainter, on the far side (light that crossed the glass),
	// in the backdrop's own colour, much more saturated; over a faint glow
	// down the bevel.
	float facing = dot(outward, -light_dir);
	float lit = 0.35 + 0.65 * pow(max(facing, 0.0), 1.5) + 0.5 * pow(max(-facing, 0.0), 1.5);
	float line = rim_share;
	vec3 behind = clamp(mix(vec3(y), backdrop, 4.0), 0.0, 1.0);
	vec3 spec = mix(vec3(1.0), behind, 0.55);
	rgb = mix(rgb, spec, clamp(highlight * lit * line, 0.0, 1.0));
	rgb += vec3(0.08 * highlight * pow(1.0 - clamp(x, 0.0, 1.0), 3.0));

	vec4 glass = vec4(min(rgb, vec3(1.0)), 1.0);
	gl_FragColor = (glass * here + vec4(0.0, 0.0, 0.0, shade) * (1.0 - here)) * alpha;
}
