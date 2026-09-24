// Liquid Glass: the blurred background behind a glass panel, bent near the
// panel's edge the way a thick lens bends it, with a little colour fringe.
// The panel's shape comes from its own buffer's alpha (the mask) when there
// is one, else from the node's rounded box.

#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif

varying vec2 v_texcoord;
uniform sampler2D tex;   // the blurred background, the whole buffer
uniform sampler2D mask;  // the panel's buffer
uniform bool has_mask;
uniform vec2 texel;      // one buffer pixel, in tex coordinates
uniform vec2 box_pos;    // the node, in buffer pixels
uniform vec2 box_size;
uniform vec4 mask_src;   // the mask's source box, normalized
uniform float radius;    // corners, without a mask
uniform float refraction; // how far light bends at the very edge, in pixels
uniform float thickness;  // how far in from the edge it bends at all
uniform float alpha;

float inside(vec2 p) {
	if (has_mask) {
		vec2 r = (p - box_pos) / box_size;
		if (r.x < 0.0 || r.y < 0.0 || r.x > 1.0 || r.y > 1.0) {
			return 0.0;
		}
		// Any pixel the panel draws is glass, however translucent: only
		// its outline is an edge, not the icons and text on it.
		return smoothstep(0.0, 0.04, texture2D(mask, mask_src.xy + r * mask_src.zw).a);
	}
	vec2 q = abs(p - box_pos - box_size * 0.5) - box_size * 0.5 + radius;
	float d = min(max(q.x, q.y), 0.0) + length(max(q, 0.0)) - radius;
	return clamp(0.5 - d, 0.0, 1.0);
}

void main() {
	vec2 p = gl_FragCoord.xy;
	vec2 grad = vec2(0.0);
	float edge = 0.0;
	// Three rings, for a smooth falloff instead of a step at `thickness`.
	for (int i = 1; i <= 3; i++) {
		float t = thickness * float(i) / 3.0;
		float l = inside(p - vec2(t, 0.0));
		float r = inside(p + vec2(t, 0.0));
		float u = inside(p - vec2(0.0, t));
		float d = inside(p + vec2(0.0, t));
		grad += vec2(r - l, d - u);
		edge += 1.0 - min(min(l, r), min(u, d));
	}
	edge /= 3.0;
	vec2 inward = length(grad) > 0.0001 ? normalize(grad) : vec2(0.0);
	vec2 off = inward * refraction * edge * edge;

	// The rim shows what lies further in, each colour bent a little apart.
	vec4 c = texture2D(tex, v_texcoord + off * texel);
	c.r = texture2D(tex, v_texcoord + off * 1.12 * texel).r;
	c.b = texture2D(tex, v_texcoord + off * 0.88 * texel).b;
	gl_FragColor = c * alpha;
}
