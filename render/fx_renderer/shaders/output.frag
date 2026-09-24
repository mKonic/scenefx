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
	gl_FragColor = vec4(e, 1.0);
}
