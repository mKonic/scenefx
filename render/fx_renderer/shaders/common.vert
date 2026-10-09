uniform mat3 proj;
uniform vec4 color;
uniform mat3 tex_proj;
attribute vec2 pos;
varying vec4 v_color;
varying vec2 v_texcoord;
// The framebuffer's size, and where a fragment is in it (see v_frag).
uniform vec2 frag_size;
varying vec2 v_frag;
// A bent texture (fx_render_texture_options.warp): each vertex says where in
// the unbent box it comes from, so texturing and the corners (v_frag) are
// worked out there.
uniform float warped;
uniform vec4 warp_box;
attribute vec2 texcoord;

void main() {
	vec3 pos3 = vec3(pos, 1.0);
	gl_Position = vec4(pos3 * proj, 1.0);
	v_color = color;
	if (warped > 0.5) {
		v_texcoord = (vec3(texcoord, 1.0) * tex_proj).xy;
		v_frag = warp_box.xy + texcoord * warp_box.zw;
	} else {
		v_texcoord = (pos3 * tex_proj).xy;
		v_frag = (gl_Position.xy * 0.5 + 0.5) * frag_size;
	}
}
