uniform mat3 proj;
uniform vec4 color;
uniform mat3 tex_proj;
attribute vec2 pos;
varying vec4 v_color;
varying vec2 v_texcoord;
// The framebuffer's size, and where a fragment is in it (see v_frag).
uniform vec2 frag_size;
varying vec2 v_frag;

void main() {
	vec3 pos3 = vec3(pos, 1.0);
	gl_Position = vec4(pos3 * proj, 1.0);
	v_color = color;
	v_texcoord = (pos3 * tex_proj).xy;
	v_frag = (gl_Position.xy * 0.5 + 0.5) * frag_size;
}
