#ifndef SCENEFX_FX_OPENGL_H
#define SCENEFX_FX_OPENGL_H

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <wlr/backend.h>
#include <wlr/render/interface.h>
#include <wlr/types/wlr_buffer.h>

struct fx_renderer;

struct wlr_renderer *fx_renderer_create_with_drm_fd(int drm_fd);
struct wlr_renderer *fx_renderer_create(struct wlr_backend *backend);

struct fx_renderer *fx_get_renderer(struct wlr_renderer *wlr_renderer);

bool fx_renderer_check_ext(struct wlr_renderer *renderer, const char *ext);
GLuint fx_renderer_get_buffer_fbo(struct wlr_renderer *renderer, struct wlr_buffer *buffer);

/**
 * Hyprland's decoration:screen_shader: every output's finished frame drawn
 * through `source`, a GLSL ES fragment shader (#version 300 es or 320 es, or
 * 1.00 without a #version) reading `tex` at `v_texcoord`, with the optional
 * uniforms `time` (seconds), `wl_output`, `screen_size` (also `fullSize`,
 * `screenSize`) and `pointer_position` (0 to 1 across the output). NULL or
 * "" turns it off. On a compile or link error it stays as it was, false is
 * returned and the error is in `error`.
 */
bool fx_renderer_set_screen_shader(struct wlr_renderer *renderer, const char *source,
	char *error, size_t error_len);
/** Whether the screen shader changes by itself (uses time or the pointer):
 * then every frame must be drawn whole. */
bool fx_renderer_screen_shader_animates(struct wlr_renderer *renderer);
/** The uniforms for the next frame of an output. */
void fx_renderer_set_screen_shader_frame(struct wlr_renderer *renderer, float time,
	int output, float pointer_x, float pointer_y);

//
// fx_texture
//

struct fx_texture_attribs {
	GLenum target; /* either GL_TEXTURE_2D or GL_TEXTURE_EXTERNAL_OES */
	GLuint tex;

	bool has_alpha;
};

struct wlr_texture *fx_texture_from_buffer(struct wlr_renderer *wlr_renderer,
		struct wlr_buffer *buffer);

void fx_texture_get_attribs(struct wlr_texture *texture,
	struct fx_texture_attribs *attribs);

#endif
