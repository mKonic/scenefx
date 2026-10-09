// The HDR output pass: the blend buffer holds the frame gamma-encoded as
// always (1.0 = SDR white; HDR content goes above it), and this turns it
// into the screen's signal: linear light, into its primaries at its
// luminance, then its transfer function (PQ for HDR10).

// PQ raises to the 78.8th power: half precision is visibly wrong.
precision highp float;

varying vec2 v_texcoord;

uniform sampler2D tex;
uniform mat3 matrix;
// 0: gamma 2.2, 1: PQ, 2: linear, 3: sRGB
uniform int out_tf;
// The display's correction table (its colour profile) on the encoded
// signal: lut_size^3 entries, each blue slice a lut_size square tile, the
// tiles side by side. 0: none.
uniform sampler2D lut;
uniform int lut_size;

vec3 lut_sample(vec3 c) {
	float n = float(lut_size);
	c = clamp(c, 0.0, 1.0) * (n - 1.0);
	float slice = floor(c.b);
	float next = min(slice + 1.0, n - 1.0);
	float t = c.b - slice;
	vec2 in_tile = vec2(c.r + 0.5, c.g + 0.5);
	vec3 a = texture2D(lut, vec2((slice * n + in_tile.x) / (n * n), in_tile.y / n)).rgb;
	vec3 b = texture2D(lut, vec2((next * n + in_tile.x) / (n * n), in_tile.y / n)).rgb;
	return mix(a, b, t);
}

vec3 pq_encode(vec3 l) {
	const float m1 = 0.1593017578125;
	const float m2 = 78.84375;
	const float c1 = 0.8359375;
	const float c2 = 18.8515625;
	const float c3 = 18.6875;
	vec3 p = pow(clamp(l, 0.0, 1.0), vec3(m1));
	return pow((c1 + c2 * p) / (1.0 + c3 * p), vec3(m2));
}

vec3 srgb_encode(vec3 l) {
	vec3 lo = l * 12.92;
	vec3 hi = 1.055 * pow(l, vec3(1.0 / 2.4)) - 0.055;
	return mix(lo, hi, step(vec3(0.0031308), l));
}

void main() {
	vec3 c = texture2D(tex, v_texcoord).rgb;
	vec3 lin = sign(c) * pow(abs(c), vec3(2.2));
	lin = max(matrix * lin, 0.0);
	vec3 e;
	if (out_tf == 1) {
		e = pq_encode(lin);
	} else if (out_tf == 2) {
		e = lin;
	} else if (out_tf == 3) {
		e = srgb_encode(min(lin, 1.0));
	} else {
		e = pow(min(lin, 1.0), vec3(1.0 / 2.2));
	}
	if (lut_size > 1) {
		e = lut_sample(e);
	}
	gl_FragColor = vec4(e, 1.0);
}
