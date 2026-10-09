#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <pixman.h>
#include <time.h>
#include <unistd.h>
#include <wlr/render/allocator.h>
#include <wlr/render/drm_syncobj.h>
#include <wlr/util/transform.h>
#include <wlr/util/log.h>
#include <wlr/util/region.h>

#include "render/color.h"
#include "render/egl.h"
#include "render/fx_renderer/fx_renderer.h"
#include "render/fx_renderer/shaders.h"
#include "render/pass.h"
#include "render/tracy.h"
#include "scenefx/render/fx_renderer/fx_offscreen_buffers.h"
#include "scenefx/render/fx_renderer/fx_renderer.h"
#include "scenefx/types/fx/blur_data.h"
#include "scenefx/types/wlr_scene.h"
#include "util/matrix.h"

#define MAX_QUADS 86 // 4kb

struct fx_render_texture_options fx_render_texture_options_default(
		const struct wlr_render_texture_options *base) {
	struct fx_render_texture_options options = {
		.corners = {0},
		.discard_transparent = false,
		.clip_box = NULL,
		.clipped_region = {0},
	};
	memcpy(&options.base, base, sizeof(*base));
	return options;
}

struct fx_render_rect_options fx_render_rect_options_default(
		const struct wlr_render_rect_options *base) {
	struct fx_render_rect_options options = {
		.base = *base,
		.clipped_region = {
			.area = { .0, .0, .0, .0 },
			.corners = {0},
		},
	};
	return options;
}

static void render(const struct wlr_box *box, const pixman_region32_t *clip, GLint attrib);
static void set_proj_matrix(GLint loc, float proj[9], const struct wlr_box *box);
struct tex_shader;
static void render_warp(const struct fx_render_texture_options *options,
		const struct wlr_box *whole, const pixman_region32_t *clip,
		const struct tex_shader *shader);
static void set_tex_matrix(GLint loc, enum wl_output_transform trans,
		const struct wlr_fbox *box);

static void wlr_matrix_transpose(float out[static 9], const float in[static 9]) {
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			out[c * 3 + r] = in[r * 3 + c];
		}
	}
}

static int output_tf_from_wlr(enum wlr_color_transfer_function tf) {
	switch (tf) {
	case WLR_COLOR_TRANSFER_FUNCTION_ST2084_PQ:
		return 1;
	case WLR_COLOR_TRANSFER_FUNCTION_EXT_LINEAR:
		return 2;
	case WLR_COLOR_TRANSFER_FUNCTION_SRGB:
		return 3;
	case WLR_COLOR_TRANSFER_FUNCTION_GAMMA22:
	case WLR_COLOR_TRANSFER_FUNCTION_BT1886:
		return 0;
	}
	return 0;
}

static void take_color_transform(struct fx_gles_render_pass *pass,
		struct wlr_color_transform *tr, bool *lut) {
	switch (tr->type) {
	case COLOR_TRANSFORM_MATRIX:;
		struct wlr_color_transform_matrix *m = wl_container_of(tr, m, base);
		wlr_matrix_multiply(pass->output_matrix, m->matrix, pass->output_matrix);
		break;
	case COLOR_TRANSFORM_INVERSE_EOTF:
		pass->output_tf = output_tf_from_wlr(wlr_color_transform_inverse_eotf_from_base(tr)->tf);
		break;
	case COLOR_TRANSFORM_PIPELINE:;
		struct wlr_color_transform_pipeline *p = wl_container_of(tr, p, base);
		for (size_t i = 0; i < p->len; i++) {
			take_color_transform(pass, p->transforms[i], lut);
		}
		break;
	case COLOR_TRANSFORM_LUT_3X1D:
	case COLOR_TRANSFORM_LCMS2:
		*lut = true;
		break;
	}
}

void fx_render_pass_set_color_transform(struct fx_gles_render_pass *pass,
		struct wlr_color_transform *transform) {
	wlr_matrix_identity(pass->output_matrix);
	pass->output_tf = 0;
	pass->two_pass = false;
	if (transform == NULL) {
		return;
	}
	bool lut = false;
	take_color_transform(pass, transform, &lut);
	if (lut) {
		// Gamma ramps and ICC profiles have no GLES path yet: drawn as if
		// they weren't there, as before this renderer took transforms.
		static bool logged = false;
		if (!logged) {
			wlr_log(WLR_INFO, "fx_renderer: output color LUTs are not applied");
			logged = true;
		}
	}
	float identity[9];
	wlr_matrix_identity(identity);
	bool plain = memcmp(pass->output_matrix, identity, sizeof(identity)) == 0 &&
		(pass->output_tf == 0 || pass->output_tf == 3);
	pass->two_pass = !plain;
}

bool fx_render_pass_init_offscreen_buffers(struct wlr_render_pass *render_pass,
		struct wlr_output *output) {
	struct fx_gles_render_pass *pass = fx_get_render_pass(render_pass);
	if (output == NULL) {
		pass->fx_offscreen_buffers = NULL;
		return false;
	}
	if (pass->fx_offscreen_buffers != NULL) {
		wlr_log(WLR_ERROR, "Extra buffers called twice. Ignoring...");
		return true;
	}

	// For per output framebuffers
	pass->fx_offscreen_buffers = fx_offscreen_buffers_try_get(output);
	if (pass->fx_offscreen_buffers == NULL) {
		wlr_log(WLR_ERROR, "Failed to get/create effect framebuffers for output: %s",
				output->name);
		return false;
	}

	// Update the buffers if needed
	struct fx_renderer *renderer = pass->buffer->renderer;
	const int width = pass->buffer->buffer->width;
	const int height = pass->buffer->buffer->height;
	bool failed = false;
	fx_framebuffer_get_or_create_custom(renderer, output->allocator, width, height, false,
			&pass->fx_offscreen_buffers->blur_saved_pixels_buffer, &failed);
	fx_framebuffer_get_or_create_custom(renderer, output->allocator, width, height, true,
			&pass->fx_offscreen_buffers->effects_buffer, &failed);
	fx_framebuffer_get_or_create_custom(renderer, output->allocator, width, height, true,
			&pass->fx_offscreen_buffers->effects_buffer_swapped, &failed);
	fx_framebuffer_get_or_create_custom(renderer, output->allocator, width, height, false,
			&pass->fx_offscreen_buffers->optimized_blur_buffer, &failed);
	fx_framebuffer_get_or_create_custom(renderer, output->allocator, width, height, false,
			&pass->fx_offscreen_buffers->optimized_no_blur_buffer, &failed);
	// A screen shader draws the finished frame: it's drawn aside first.
	if (renderer->screen_shader.program) {
		pass->two_pass = true;
		fx_framebuffer_get_or_create_half_float(renderer, width, height,
				&pass->fx_offscreen_buffers->screen_shader_buffer, &failed);
	}
	if (pass->two_pass) {
		fx_framebuffer_get_or_create_half_float(renderer, width, height,
				&pass->fx_offscreen_buffers->blend_buffer, &failed);
	}

	if (failed) {
		fx_framebuffer_bind(pass->buffer);
		fx_offscreen_buffers_destroy(pass->fx_offscreen_buffers);
		pass->fx_offscreen_buffers = NULL;
		pass->two_pass = false;
		wlr_log(WLR_ERROR, "Failed to create effect framebuffers");
		return false;
	}

	// From here on the frame is drawn into the blend buffer; submit
	// converts it into the output's buffer.
	if (pass->two_pass) {
		pass->output_buffer = pass->buffer;
		pass->buffer = pass->fx_offscreen_buffers->blend_buffer;
	}

	// Bind back to the default buffer
	fx_framebuffer_bind(pass->buffer);
	return true;
}

// The finished frame through the screen shader, into the buffer the output
// pass then reads.
static struct fx_framebuffer *render_screen_shader(struct fx_gles_render_pass *pass,
		struct fx_framebuffer *frame) {
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct fx_framebuffer *out = pass->fx_offscreen_buffers ?
		pass->fx_offscreen_buffers->screen_shader_buffer : NULL;
	if (!renderer->screen_shader.program || out == NULL) {
		return frame;
	}
	const int width = frame->buffer->width, height = frame->buffer->height;
	fx_framebuffer_bind(out);
	glViewport(0, 0, width, height);
	glDisable(GL_BLEND);
	glDisable(GL_STENCIL_TEST);
	glDisable(GL_SCISSOR_TEST);

	glUseProgram(renderer->screen_shader.program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, frame->tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glUniform1i(renderer->screen_shader.tex, 0);
	glUniform1f(renderer->screen_shader.time, renderer->screen_shader.time_value);
	glUniform1i(renderer->screen_shader.wl_output, renderer->screen_shader.output_value);
	glUniform2f(renderer->screen_shader.screen_size, width, height);
	glUniform2f(renderer->screen_shader.pointer, renderer->screen_shader.pointer_x,
		renderer->screen_shader.pointer_y);
	struct wlr_box box = { 0, 0, width, height };
	set_proj_matrix(renderer->screen_shader.proj, pass->projection_matrix, &box);
	render(&box, NULL, renderer->screen_shader.pos_attrib);
	glBindTexture(GL_TEXTURE_2D, 0);
	glEnable(GL_BLEND);
	return out;
}

static void render_output_pass(struct fx_gles_render_pass *pass) {
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct output_shader *shader = &renderer->shaders.output;
	struct fx_framebuffer *blend = render_screen_shader(pass, pass->buffer);
	const int width = pass->output_buffer->buffer->width;
	const int height = pass->output_buffer->buffer->height;

	fx_framebuffer_bind(pass->output_buffer);
	glViewport(0, 0, width, height);
	glDisable(GL_BLEND);
	glDisable(GL_STENCIL_TEST);
	glDisable(GL_SCISSOR_TEST);

	glUseProgram(shader->program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, blend->tex);
	glUniform1i(shader->tex, 0);
	// wlroots matrices are row-major; GLSL reads column-major.
	float transposed[9];
	wlr_matrix_transpose(transposed, pass->output_matrix);
	glUniformMatrix3fv(shader->matrix, 1, GL_FALSE, transposed);
	glUniform1i(shader->out_tf, pass->output_tf);

	struct wlr_box box = { 0, 0, width, height };
	struct wlr_fbox src = { 0, 0, 1, 1 };
	set_proj_matrix(shader->proj, pass->projection_matrix, &box);
	set_tex_matrix(shader->tex_proj, WL_OUTPUT_TRANSFORM_NORMAL, &src);
	render(&box, NULL, shader->pos_attrib);

	glBindTexture(GL_TEXTURE_2D, 0);
	glEnable(GL_BLEND);

	// Say what these pixels are, so copies (screenshots, screen sharing)
	// can turn them back into what SDR content looks like.
	struct fx_framebuffer *out = pass->output_buffer;
	out->encoded_tf = 0;
	if (pass->output_tf == 1 || pass->output_tf == 2) {
		matrix_invert(out->encoded_matrix, pass->output_matrix);
		out->encoded_tf = pass->output_tf;
	}

	pass->buffer = pass->output_buffer;
	pass->output_buffer = NULL;
}

///
/// Base Wlroots pass functions
///

static const struct wlr_render_pass_impl render_pass_impl;

struct fx_gles_render_pass *fx_get_render_pass(struct wlr_render_pass *render_pass) {
	assert(render_pass->impl == &render_pass_impl);
	struct fx_gles_render_pass *pass = wl_container_of(render_pass, pass, base);
	return pass;
}

static bool render_pass_submit(struct wlr_render_pass *wlr_pass) {
	struct fx_gles_render_pass *pass = fx_get_render_pass(wlr_pass);
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct fx_render_timer *timer = pass->timer;
	bool ok = false;

	TRACY_BOTH_ZONES_START(pass->buffer->renderer);
	push_fx_debug(renderer);

	if (pass->two_pass && pass->output_buffer != NULL) {
		render_output_pass(pass);
	}

	if (timer) {
		// clear disjoint flag
		GLint64 disjoint;
		renderer->procs.glGetInteger64vEXT(GL_GPU_DISJOINT_EXT, &disjoint);
		// set up the query
		renderer->procs.glQueryCounterEXT(timer->id, GL_TIMESTAMP_EXT);
		// get end-of-CPU-work time in GL time domain
		renderer->procs.glGetInteger64vEXT(GL_TIMESTAMP_EXT, &timer->gl_cpu_end);
		// get end-of-CPU-work time in CPU time domain
		clock_gettime(CLOCK_MONOTONIC, &timer->cpu_end);
	}

	if (pass->signal_timeline != NULL) {
		EGLSyncKHR sync = wlr_egl_create_sync(renderer->egl, -1);
		if (sync == EGL_NO_SYNC_KHR) {
			goto out;
		}

		int sync_file_fd = wlr_egl_dup_fence_fd(renderer->egl, sync);
		wlr_egl_destroy_sync(renderer->egl, sync);
		if (sync_file_fd < 0) {
			goto out;
		}

		ok = wlr_drm_syncobj_timeline_import_sync_file(pass->signal_timeline, pass->signal_point, sync_file_fd);
		close(sync_file_fd);
		if (!ok) {
			goto out;
		}
	} else {
		glFlush();
	}

	ok = true;

out:
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
	TRACY_GPU_ZONE_COLLECT(renderer);

	wlr_egl_restore_context(&pass->prev_ctx);

	wlr_drm_syncobj_timeline_unref(pass->signal_timeline);
	wlr_buffer_unlock(pass->buffer->buffer);

	pass->fx_offscreen_buffers = NULL;
	pixman_region32_fini(&pass->blur_padding_region);

	free(pass);

	return ok;
}

static void render_pass_add_texture(struct wlr_render_pass *wlr_pass,
		const struct wlr_render_texture_options *options) {
	struct fx_gles_render_pass *pass = fx_get_render_pass(wlr_pass);
	const struct fx_render_texture_options fx_options =
		fx_render_texture_options_default(options);
	// Re-use fx function but with default options
	// TODO: Simplified version?
	fx_render_pass_add_texture(pass, &fx_options);
}

static void render_pass_add_rect(struct wlr_render_pass *wlr_pass,
		const struct wlr_render_rect_options *options) {
	struct fx_gles_render_pass *pass = fx_get_render_pass(wlr_pass);
	const struct fx_render_rect_options fx_options =
		fx_render_rect_options_default(options);
	// Re-use fx function but with default options
	// TODO: Simplified version?
	fx_render_pass_add_rect(pass, &fx_options);
}

static const struct wlr_render_pass_impl render_pass_impl = {
	.submit = render_pass_submit,
	.add_texture = render_pass_add_texture,
	.add_rect = render_pass_add_rect,
};

///
/// FX pass functions
///

// TODO: REMOVE STENCILING

// Initialize the stenciling work
static void stencil_mask_init(void) {
	glClearStencil(0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glEnable(GL_STENCIL_TEST);

	glStencilFunc(GL_ALWAYS, 1, 0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
	// Disable writing to color buffer
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
}

// Close the mask
static void stencil_mask_close(bool draw_inside_mask) {
	// Reenable writing to color buffer
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	if (draw_inside_mask) {
		glStencilFunc(GL_EQUAL, 1, 0xFF);
		glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
		return;
	}
	glStencilFunc(GL_NOTEQUAL, 1, 0xFF);
	glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
}

// Finish stenciling and clear the buffer
static void stencil_mask_fini(void) {
	glClearStencil(0);
	glClear(GL_STENCIL_BUFFER_BIT);
	glDisable(GL_STENCIL_TEST);
}

static void render(const struct wlr_box *box, const pixman_region32_t *clip, GLint attrib) {
	// Where fragments are, in highp (see common.vert's v_frag): the size of
	// what is drawn into.
	GLint program = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &program);
	GLint frag_size = program ? glGetUniformLocation((GLuint)program, "frag_size") : -1;
	if (frag_size >= 0) {
		GLint viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);
		glUniform2f(frag_size, (GLfloat)viewport[2], (GLfloat)viewport[3]);
	}

	pixman_region32_t region;
	pixman_region32_init_rect(&region, box->x, box->y, box->width, box->height);

	if (clip) {
		pixman_region32_intersect(&region, &region, clip);
	}

	int rects_len;
	const pixman_box32_t *rects = pixman_region32_rectangles(&region, &rects_len);
	if (rects_len == 0) {
		pixman_region32_fini(&region);
		return;
	}

	glEnableVertexAttribArray(attrib);

	for (int i = 0; i < rects_len;) {
		int batch = rects_len - i < MAX_QUADS ? rects_len - i : MAX_QUADS;
		int batch_end = batch + i;

		size_t vert_index = 0;
		GLfloat verts[MAX_QUADS * 6 * 2];
		for (; i < batch_end; i++) {
			const pixman_box32_t *rect = &rects[i];

			verts[vert_index++] = (GLfloat)(rect->x1 - box->x) / box->width;
			verts[vert_index++] = (GLfloat)(rect->y1 - box->y) / box->height;
			verts[vert_index++] = (GLfloat)(rect->x2 - box->x) / box->width;
			verts[vert_index++] = (GLfloat)(rect->y1 - box->y) / box->height;
			verts[vert_index++] = (GLfloat)(rect->x1 - box->x) / box->width;
			verts[vert_index++] = (GLfloat)(rect->y2 - box->y) / box->height;
			verts[vert_index++] = (GLfloat)(rect->x2 - box->x) / box->width;
			verts[vert_index++] = (GLfloat)(rect->y1 - box->y) / box->height;
			verts[vert_index++] = (GLfloat)(rect->x2 - box->x) / box->width;
			verts[vert_index++] = (GLfloat)(rect->y2 - box->y) / box->height;
			verts[vert_index++] = (GLfloat)(rect->x1 - box->x) / box->width;
			verts[vert_index++] = (GLfloat)(rect->y2 - box->y) / box->height;
		}

		glVertexAttribPointer(attrib, 2, GL_FLOAT, GL_FALSE, 0, verts);
		glDrawArrays(GL_TRIANGLES, 0, batch * 6);
	}

	glDisableVertexAttribArray(attrib);

	pixman_region32_fini(&region);
}

// A bent texture's grid as triangles, drawn once per clip rectangle with the
// rest scissored away (the render buffer's coordinates are GL's: y = 0 is
// its first row).
static void render_warp(const struct fx_render_texture_options *options,
		const struct wlr_box *whole, const pixman_region32_t *clip,
		const struct tex_shader *shader) {
	int rects_len;
	const pixman_box32_t *rects = pixman_region32_rectangles(clip, &rects_len);
	if (rects_len == 0 || shader->texcoord_attrib < 0) {
		return;
	}

	int cols = options->warp_cols, rows = options->warp_rows;
	size_t verts_len = (size_t)cols * rows * 6;
	GLfloat *pos = malloc(verts_len * 2 * sizeof(GLfloat));
	GLfloat *uv = malloc(verts_len * 2 * sizeof(GLfloat));
	if (pos == NULL || uv == NULL) {
		free(pos);
		free(uv);
		return;
	}
	static const int corner[6][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 1}};
	size_t n = 0;
	for (int j = 0; j < rows; j++) {
		for (int i = 0; i < cols; i++) {
			for (int c = 0; c < 6; c++) {
				const float *p = &options->warp[
					((size_t)(j + corner[c][1]) * (cols + 1) + i + corner[c][0]) * 4];
				pos[n * 2] = p[0] / whole->width;
				pos[n * 2 + 1] = p[1] / whole->height;
				uv[n * 2] = p[2];
				uv[n * 2 + 1] = p[3];
				n++;
			}
		}
	}

	GLint program = 0;
	glGetIntegerv(GL_CURRENT_PROGRAM, &program);
	GLint frag_size = glGetUniformLocation((GLuint)program, "frag_size");
	if (frag_size >= 0) {
		glUniform2f(frag_size, whole->width, whole->height);
	}
	glUniform1f(shader->warped, 1.0f);
	glUniform4f(shader->warp_box, options->warp_box.x, options->warp_box.y,
		options->warp_box.width, options->warp_box.height);

	glEnableVertexAttribArray(shader->pos_attrib);
	glEnableVertexAttribArray(shader->texcoord_attrib);
	glVertexAttribPointer(shader->pos_attrib, 2, GL_FLOAT, GL_FALSE, 0, pos);
	glVertexAttribPointer(shader->texcoord_attrib, 2, GL_FLOAT, GL_FALSE, 0, uv);
	glEnable(GL_SCISSOR_TEST);
	for (int i = 0; i < rects_len; i++) {
		const pixman_box32_t *r = &rects[i];
		glScissor(r->x1, r->y1, r->x2 - r->x1, r->y2 - r->y1);
		glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts_len);
	}
	glDisable(GL_SCISSOR_TEST);
	glDisableVertexAttribArray(shader->texcoord_attrib);
	glDisableVertexAttribArray(shader->pos_attrib);
	glUniform1f(shader->warped, 0.0f);

	free(pos);
	free(uv);
}

static void set_proj_matrix(GLint loc, float proj[9], const struct wlr_box *box) {
	float gl_matrix[9];
	wlr_matrix_identity(gl_matrix);
	wlr_matrix_translate(gl_matrix, box->x, box->y);
	wlr_matrix_scale(gl_matrix, box->width, box->height);
	wlr_matrix_multiply(gl_matrix, proj, gl_matrix);
	glUniformMatrix3fv(loc, 1, GL_FALSE, gl_matrix);
}

static void set_tex_matrix(GLint loc, enum wl_output_transform trans,
		const struct wlr_fbox *box) {
	float tex_matrix[9];
	wlr_matrix_identity(tex_matrix);
	wlr_matrix_translate(tex_matrix, box->x, box->y);
	wlr_matrix_scale(tex_matrix, box->width, box->height);
	wlr_matrix_translate(tex_matrix, .5, .5);

	// since textures have a different origin point we have to transform
	// differently if we are rotating
	if (trans & WL_OUTPUT_TRANSFORM_90) {
		wlr_matrix_transform(tex_matrix, wlr_output_transform_invert(trans));
	} else {
		wlr_matrix_transform(tex_matrix, trans);
	}
	wlr_matrix_translate(tex_matrix, -.5, -.5);

	glUniformMatrix3fv(loc, 1, GL_FALSE, tex_matrix);
}

static void setup_blending(enum wlr_render_blend_mode mode) {
	switch (mode) {
	case WLR_RENDER_BLEND_MODE_PREMULTIPLIED:
		glEnable(GL_BLEND);
		break;
	case WLR_RENDER_BLEND_MODE_NONE:
		glDisable(GL_BLEND);
		break;
	}
}

static bool apply_clip_region(pixman_region32_t *clip_region,
		const struct wlr_box *clipped_region_box, const struct fx_corner_fradii *corners) {
	if (!wlr_box_empty(clipped_region_box)) {
		float top = fmax(corners->top_left, corners->top_right);
		float bottom = fmax(corners->bottom_left, corners->bottom_right);
		float left = fmax(corners->top_left, corners->bottom_left);
		float right = fmax(corners->top_right, corners->bottom_right);

		pixman_region32_t user_clip_region;
		pixman_region32_init_rect(
			&user_clip_region,
			clipped_region_box->x + (left * 0.3),
			clipped_region_box->y + (top * 0.3),
			fmax(clipped_region_box->width - (left + right) * 0.3, 0),
			fmax(clipped_region_box->height - (top + bottom) * 0.3, 0)
		);
		pixman_region32_subtract(clip_region, clip_region, &user_clip_region);
		pixman_region32_fini(&user_clip_region);
		return true;
	}

	return false;
}

void fx_render_pass_add_texture(struct fx_gles_render_pass *pass,
		const struct fx_render_texture_options *fx_options) {
	const struct wlr_render_texture_options *options = &fx_options->base;
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct fx_texture *texture = fx_get_texture(options->texture);

	struct tex_shader *shader = NULL;

	bool use_effects = !fx_corner_fradii_is_empty(&fx_options->corners)
		|| clipped_fregion_is_valid(&fx_options->clipped_region);
	switch (texture->target) {
	case GL_TEXTURE_2D:
		if (texture->has_alpha) {
			shader = use_effects
				? &renderer->shaders.tex_effects_rgba
				: &renderer->shaders.tex_rgba;
		} else {
			shader = use_effects
				? &renderer->shaders.tex_effects_rgbx
				: &renderer->shaders.tex_rgbx;
		}
		break;
	case GL_TEXTURE_EXTERNAL_OES:
		// EGL_EXT_image_dma_buf_import_modifiers requires
		// GL_OES_EGL_image_external
		assert(renderer->exts.OES_egl_image_external);
		shader = use_effects
			? &renderer->shaders.tex_effects_ext
			: &renderer->shaders.tex_ext;
		break;
	default:
		abort();
	}

	struct wlr_box dst_box;
	struct wlr_fbox src_fbox;
	wlr_render_texture_options_get_src_box(options, &src_fbox);
	wlr_render_texture_options_get_dst_box(options, &dst_box);
	float alpha = wlr_render_texture_options_get_alpha(options);

	const struct wlr_box *clip_box = &dst_box;
	if (!wlr_box_empty(fx_options->clip_box)) {
		clip_box = fx_options->clip_box;
	}

	src_fbox.x /= options->texture->width;
	src_fbox.y /= options->texture->height;
	src_fbox.width /= options->texture->width;
	src_fbox.height /= options->texture->height;

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("dst_box (WxH, X, Y): %dx%d, %d, %d",
			dst_box.width, dst_box.height, dst_box.x, dst_box.y);
	TRACY_ZONE_TEXT_f("clip_box (WxH, X, Y): %dx%d, %d, %d",
			clip_box->width, clip_box->height, clip_box->x, clip_box->y);
	TRACY_ZONE_TEXT_f("src_box (WxH, X, Y): %lfx%lf, %lf, %lf",
			src_fbox.width, src_fbox.height, src_fbox.x, src_fbox.y);
	TRACY_ZONE_TEXT_f("Shader Type: %s",
			use_effects ? (
			 shader == &renderer->shaders.tex_effects_rgba ? "Effects RGBA"
			 : shader == &renderer->shaders.tex_effects_rgbx ? "Effects RGBX"
			 : "Effects EXT"
			) : (
				shader == &renderer->shaders.tex_rgba ? "RGBA"
				: shader == &renderer->shaders.tex_rgbx ? "RGBX"
				: "EXT"
			)
		);
	push_fx_debug(renderer);

	if (options->wait_timeline != NULL) {
		int sync_file_fd =
			wlr_drm_syncobj_timeline_export_sync_file(options->wait_timeline, options->wait_point);
		if (sync_file_fd < 0) {
			TRACY_BOTH_ZONES_END_FAIL;
			return;
		}

		EGLSyncKHR sync = wlr_egl_create_sync(renderer->egl, sync_file_fd);
		close(sync_file_fd);
		if (sync == EGL_NO_SYNC_KHR) {
			TRACY_BOTH_ZONES_END_FAIL;
			return;
		}

		bool ok = wlr_egl_wait_sync(renderer->egl, sync);
		wlr_egl_destroy_sync(renderer->egl, sync);
		if (!ok) {
			TRACY_BOTH_ZONES_END_FAIL;
			return;
		}
	}

	bool has_alpha = texture->has_alpha || alpha < 1.0 || use_effects
		|| fx_options->warp != NULL || fx_options->motion_samples > 1;
	TRACY_ZONE_TEXT_f("Has Alpha: %d", has_alpha);
	setup_blending(!has_alpha ? WLR_RENDER_BLEND_MODE_NONE : options->blend_mode);

	pixman_region32_t clip_region;
	if (options->clip) {
		pixman_region32_init(&clip_region);
		pixman_region32_copy(&clip_region, options->clip);
	} else {
		pixman_region32_init_rect(&clip_region, dst_box.x, dst_box.y, dst_box.width, dst_box.height);
	}
	const struct wlr_box clipped_region_box = fx_options->clipped_region.area;
	struct fx_corner_fradii clipped_region_corners = fx_options->clipped_region.corners;
	apply_clip_region(&clip_region, &clipped_region_box, &clipped_region_corners);

	glUseProgram(shader->program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex);

	switch (options->filter_mode) {
	case WLR_SCALE_FILTER_BILINEAR:
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case WLR_SCALE_FILTER_NEAREST:
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		break;
	}

	glUniform1i(shader->tex, 0);
	glUniform1f(shader->alpha, alpha);

	glUniform1f(shader->discard_transparent, fx_options->discard_transparent);
	glUniform1f(shader->discard_below, fx_options->discard_below);

	// Content in another transfer function or gamut (HDR) is converted to
	// what the frame is drawn in; plain SDR is drawn as is.
	int hdr_tf = 0;
	switch (options->transfer_function) {
	case WLR_COLOR_TRANSFER_FUNCTION_ST2084_PQ:
		hdr_tf = 1;
		break;
	case WLR_COLOR_TRANSFER_FUNCTION_EXT_LINEAR:
		hdr_tf = 2;
		break;
	default:
		break;
	}
	float prim[9];
	float hdr_lum = options->luminance_multiplier != NULL ? *options->luminance_multiplier : 1.0f;
	wlr_matrix_identity(prim);
	if (options->transfer_function == 0 && texture->buffer != NULL && texture->buffer->encoded_tf != 0) {
		// A copy of an HDR screen's buffer (a screenshot): back to SDR.
		hdr_tf = texture->buffer->encoded_tf;
		memcpy(prim, texture->buffer->encoded_matrix, sizeof(prim));
		hdr_lum = 1.0f;
	} else if (options->primaries != NULL) {
		struct wlr_color_primaries srgb;
		wlr_color_primaries_from_named(&srgb, WLR_COLOR_NAMED_PRIMARIES_SRGB);
		wlr_color_primaries_transform_absolute_colorimetric(options->primaries, &srgb, prim);
		float identity[9];
		wlr_matrix_identity(identity);
		if (hdr_tf == 0 && memcmp(prim, identity, sizeof(prim)) != 0) {
			hdr_tf = 3; // SDR in another gamut: decoded as gamma 2.2
		}
	}
	float transposed_prim[9];
	wlr_matrix_transpose(transposed_prim, prim);
	glUniform1i(shader->hdr_tf, hdr_tf);
	glUniformMatrix3fv(shader->hdr_prim, 1, GL_FALSE, transposed_prim);
	glUniform1f(shader->hdr_lum, hdr_lum);

	if (use_effects) {
		struct fx_corner_fradii corners = fx_options->corners;

		glUniform2f(shader->effects.size, clip_box->width, clip_box->height);
		glUniform2f(shader->effects.position, clip_box->x, clip_box->y);
		uniform_corner_radii_set(&shader->effects.radius, &corners);

		glUniform2f(shader->effects.clip_size, clipped_region_box.width, clipped_region_box.height);
		glUniform2f(shader->effects.clip_position, clipped_region_box.x, clipped_region_box.y);
		uniform_corner_radii_set(&shader->effects.clip_radius, &clipped_region_corners);
	}

	set_tex_matrix(shader->tex_proj, options->transform, &src_fbox);

	glUniform1i(shader->material, fx_options->material);
	if (fx_options->material) {
		const struct wlr_box *m = &fx_options->material_box;
		glUniform4f(shader->material_box, m->x, m->y, m->width, m->height);
		glUniform2f(shader->material_texel, 1.0f / options->texture->width, 1.0f / options->texture->height);
	}
	glUniform1i(shader->motion_samples, fx_options->motion_samples);
	if (fx_options->motion_samples > 1) {
		const struct wlr_fbox *m = &fx_options->motion_box;
		glUniform4f(shader->motion_box, m->x, m->y, m->width, m->height);
		glUniform2f(shader->motion_back, fx_options->motion_back_x, fx_options->motion_back_y);
	}

	if (fx_options->warp != NULL) {
		struct wlr_box whole = {
			.width = pass->buffer->buffer->width,
			.height = pass->buffer->buffer->height,
		};
		if (use_effects) {
			glUniform2f(shader->effects.size, fx_options->warp_box.width,
				fx_options->warp_box.height);
			glUniform2f(shader->effects.position, fx_options->warp_box.x,
				fx_options->warp_box.y);
		}
		set_proj_matrix(shader->proj, pass->projection_matrix, &whole);
		render_warp(fx_options, &whole, &clip_region, shader);
	} else {
		set_proj_matrix(shader->proj, pass->projection_matrix, &dst_box);
		render(&dst_box, &clip_region, shader->pos_attrib);
	}
	pixman_region32_fini(&clip_region);

	glBindTexture(texture->target, 0);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

void fx_render_pass_add_rect(struct fx_gles_render_pass *pass,
		const struct fx_render_rect_options *fx_options) {
	const struct wlr_render_rect_options *options = &fx_options->base;

	struct fx_renderer *renderer = pass->buffer->renderer;

	const struct wlr_render_color *color = &options->color;
	struct wlr_box box;
	struct wlr_buffer *wlr_buffer = pass->buffer->buffer;
	wlr_render_rect_options_get_box(options, wlr_buffer, &box);

	const bool should_clip = clipped_fregion_is_valid(&fx_options->clipped_region);

	enum wlr_render_blend_mode blend_mode =
		(color->a == 1.0 && !should_clip) ? WLR_RENDER_BLEND_MODE_NONE : options->blend_mode;
	const bool use_fast_clear =
		blend_mode == WLR_RENDER_BLEND_MODE_NONE && // includes check for `should_clip`
		options->clip == NULL &&
		box.x == 0 && box.y == 0 &&
		box.width == wlr_buffer->width &&
		box.height == wlr_buffer->height;

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("Box (WxH, X, Y): %dx%d, %d, %d", box.width, box.height, box.x, box.y);
	TRACY_ZONE_TEXT_f("Color RGBA: %f, %f, %f, %f", color->r, color->g, color->b, color->a);
	TRACY_ZONE_TEXT_f("Use fast clear optimization: %d", use_fast_clear);

	push_fx_debug(renderer);
	if (use_fast_clear) {
		glClearColor(color->r, color->g, color->b, color->a);
		glClear(GL_COLOR_BUFFER_BIT);
	} else {
		const struct wlr_box *clipped_region_box = &fx_options->clipped_region.area;
		const struct fx_corner_fradii *clipped_region_corners = &fx_options->clipped_region.corners;

		TRACY_ZONE_TEXT_f("Clip Box (WxH, X, Y): %dx%d, %d, %d",
				clipped_region_box->width, clipped_region_box->height,
				clipped_region_box->x, clipped_region_box->y);
		TRACY_ZONE_TEXT_f("Clip Box Corners (TL, TR, BL, BR): %f, %f, %f, %f",
				clipped_region_corners->top_left,
				clipped_region_corners->top_right,
				clipped_region_corners->bottom_left,
				clipped_region_corners->bottom_right);

		pixman_region32_t clip_region;
		if (options->clip) {
			pixman_region32_init(&clip_region);
			pixman_region32_copy(&clip_region, options->clip);
		} else {
			pixman_region32_init_rect(&clip_region, box.x, box.y, box.width, box.height);
		}

		apply_clip_region(&clip_region, clipped_region_box, clipped_region_corners);

		setup_blending(blend_mode);
		struct quad_shader *shader = should_clip
			? &renderer->shaders.quad_clip
			: &renderer->shaders.quad;
		glUseProgram(shader->program);
		set_proj_matrix(shader->proj, pass->projection_matrix, &box);
		glUniform4f(shader->color, color->r, color->g, color->b, color->a);
		if (should_clip) {
			glUniform2f(shader->effects.clip_size, clipped_region_box->width, clipped_region_box->height);
			glUniform2f(shader->effects.clip_position, clipped_region_box->x, clipped_region_box->y);
			uniform_corner_radii_set(&shader->effects.clip_radius, clipped_region_corners);
		}
		render(&box, &clip_region, shader->pos_attrib);

		pixman_region32_fini(&clip_region);
	}

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

void fx_render_pass_add_rect_grad(struct fx_gles_render_pass *pass,
		const struct fx_render_rect_grad_options *fx_options) {
	const struct wlr_render_rect_options *options = &fx_options->base;

	struct fx_renderer *renderer = pass->buffer->renderer;

	if (renderer->shaders.quad_grad.max_len <= fx_options->gradient.count) {
		glDeleteProgram(renderer->shaders.quad_grad.program);
		if (!link_quad_grad_program(&renderer->shaders.quad_grad, fx_options->gradient.count + 1)) {
			wlr_log(WLR_ERROR, "Could not link quad shader after updating max_len to %d. Aborting renderer", fx_options->gradient.count + 1);
			abort();
		}
	}

	struct wlr_box box;
	struct wlr_buffer *wlr_buffer = pass->buffer->buffer;
	wlr_render_rect_options_get_box(options, wlr_buffer, &box);

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("Box (WxH, X, Y): %dx%d, %d, %d", box.width, box.height, box.x, box.y);
	TRACY_ZONE_TEXT_f("Gradient:");
	TRACY_ZONE_TEXT_f("\tNum Colors: %d", fx_options->gradient.count);
	TRACY_ZONE_TEXT_f("\tBlend: %d", fx_options->gradient.blend);
	TRACY_ZONE_TEXT_f("\tDegree: %f", fx_options->gradient.degree);
	TRACY_ZONE_TEXT_f("\tType: %s",
			fx_options->gradient.linear == 1 ? "Linear"
			: fx_options->gradient.linear == 2 ? "Conic"
			: "Unknown");
	TRACY_ZONE_TEXT_f("\tOrigin: %fx%f",
			fx_options->gradient.origin[0], fx_options->gradient.origin[1]);
	TRACY_ZONE_TEXT_f("\tRange (WxH, X, Y): %dx%d, %d, %d",
			fx_options->gradient.range.width, fx_options->gradient.range.height,
			fx_options->gradient.range.x, fx_options->gradient.range.y);
	// TODO: Display Colors (not really sure how it works without a scene example...)
	push_fx_debug(renderer);

	setup_blending(options->blend_mode);

	struct quad_grad_shader shader = renderer->shaders.quad_grad;
	glUseProgram(shader.program);

	set_proj_matrix(shader.proj, pass->projection_matrix, &box);
	glUniform4fv(shader.colors, fx_options->gradient.count, (GLfloat*)fx_options->gradient.colors);
	glUniform1i(shader.count, fx_options->gradient.count);
	glUniform2f(shader.size, fx_options->gradient.range.width, fx_options->gradient.range.height);
	glUniform1f(shader.degree, fx_options->gradient.degree);
	glUniform1f(shader.linear, fx_options->gradient.linear);
	glUniform1f(shader.blend, fx_options->gradient.blend);
	glUniform2f(shader.grad_box, fx_options->gradient.range.x, fx_options->gradient.range.y);
	glUniform2f(shader.origin, fx_options->gradient.origin[0], fx_options->gradient.origin[1]);

	render(&box, options->clip, shader.pos_attrib);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

void fx_render_pass_add_rounded_rect(struct fx_gles_render_pass *pass,
		const struct fx_render_rounded_rect_options *fx_options) {
	const struct wlr_render_rect_options *options = &fx_options->base;

	struct fx_renderer *renderer = pass->buffer->renderer;

	const struct wlr_render_color *color = &options->color;
	struct wlr_box box;
	struct wlr_buffer *wlr_buffer = pass->buffer->buffer;
	wlr_render_rect_options_get_box(options, wlr_buffer, &box);

	pixman_region32_t clip_region;
	if (options->clip) {
		pixman_region32_init(&clip_region);
		pixman_region32_copy(&clip_region, options->clip);
	} else {
		pixman_region32_init_rect(&clip_region, box.x, box.y, box.width, box.height);
	}
	const struct wlr_box *clipped_region_box = &fx_options->clipped_region.area;
	const struct fx_corner_fradii *clipped_region_corners = &fx_options->clipped_region.corners;
	apply_clip_region(&clip_region, clipped_region_box, clipped_region_corners);

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("Box (WxH, X, Y): %dx%d, %d, %d", box.width, box.height, box.x, box.y);
	TRACY_ZONE_TEXT_f("Clip Box (WxH, X, Y): %dx%d, %d, %d",
			clipped_region_box->width, clipped_region_box->height,
			clipped_region_box->x, clipped_region_box->y);
	TRACY_ZONE_TEXT_f("Clip Box Corners (TL, TR, BL, BR): %f, %f, %f, %f",
			clipped_region_corners->top_left,
			clipped_region_corners->top_right,
			clipped_region_corners->bottom_left,
			clipped_region_corners->bottom_right);
	TRACY_ZONE_TEXT_f("Color RGBA: %f, %f, %f, %f", color->r, color->g, color->b, color->a);
	TRACY_ZONE_TEXT_f("Corners (TL, TR, BL, BR): %f, %f, %f, %f",
			clipped_region_corners->top_left,
			clipped_region_corners->top_right,
			clipped_region_corners->bottom_left,
			clipped_region_corners->bottom_right);
	push_fx_debug(renderer);

	setup_blending(WLR_RENDER_BLEND_MODE_PREMULTIPLIED);

	struct quad_round_shader shader = renderer->shaders.quad_round;

	glUseProgram(shader.program);

	set_proj_matrix(shader.proj, pass->projection_matrix, &box);
	glUniform4f(shader.color, color->r, color->g, color->b, color->a);

	glUniform2f(shader.size, box.width, box.height);
	glUniform2f(shader.position, box.x, box.y);
	glUniform2f(shader.clip_size, clipped_region_box->width, clipped_region_box->height);
	glUniform2f(shader.clip_position, clipped_region_box->x, clipped_region_box->y);
	uniform_corner_radii_set(&shader.clip_radius, clipped_region_corners);

	struct fx_corner_fradii corners = fx_options->corners;
	uniform_corner_radii_set(&shader.radius, &corners);

	render(&box, &clip_region, renderer->shaders.quad_round.pos_attrib);
	pixman_region32_fini(&clip_region);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

void fx_render_pass_add_rounded_rect_grad(struct fx_gles_render_pass *pass,
		const struct fx_render_rounded_rect_grad_options *fx_options) {
	const struct wlr_render_rect_options *options = &fx_options->base;

	struct fx_renderer *renderer = pass->buffer->renderer;

	if (renderer->shaders.quad_grad_round.max_len <= fx_options->gradient.count) {
		glDeleteProgram(renderer->shaders.quad_grad_round.program);
		if (!link_quad_grad_round_program(&renderer->shaders.quad_grad_round, fx_options->gradient.count + 1)) {
			wlr_log(WLR_ERROR, "Could not link quad shader after updating max_len to %d. Aborting renderer", fx_options->gradient.count + 1);
			abort();
		}
	}

	struct wlr_box box;
	struct wlr_buffer *wlr_buffer = pass->buffer->buffer;
	wlr_render_rect_options_get_box(options, wlr_buffer, &box);

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("Box (WxH, X, Y): %dx%d, %d, %d", box.width, box.height, box.x, box.y);
	TRACY_ZONE_TEXT_f("Corners (TL, TR, BL, BR): %f, %f, %f, %f",
			fx_options->corners.top_left,
			fx_options->corners.top_right,
			fx_options->corners.bottom_left,
			fx_options->corners.bottom_right);
	TRACY_ZONE_TEXT_f("Gradient:");
	TRACY_ZONE_TEXT_f("\tNum Colors: %d", fx_options->gradient.count);
	TRACY_ZONE_TEXT_f("\tBlend: %d", fx_options->gradient.blend);
	TRACY_ZONE_TEXT_f("\tDegree: %f", fx_options->gradient.degree);
	TRACY_ZONE_TEXT_f("\tType: %s",
			fx_options->gradient.linear == 1 ? "Linear"
			: fx_options->gradient.linear == 2 ? "Conic"
			: "Unknown");
	TRACY_ZONE_TEXT_f("\tOrigin: %fx%f",
			fx_options->gradient.origin[0], fx_options->gradient.origin[1]);
	TRACY_ZONE_TEXT_f("\tRange (WxH, X, Y): %dx%d, %d, %d",
			fx_options->gradient.range.width, fx_options->gradient.range.height,
			fx_options->gradient.range.x, fx_options->gradient.range.y);
	// TODO: Display Colors (not really sure how it works without a scene example...)
	push_fx_debug(renderer);

	setup_blending(WLR_RENDER_BLEND_MODE_PREMULTIPLIED);

	struct quad_grad_round_shader shader = renderer->shaders.quad_grad_round;
	glUseProgram(shader.program);

	set_proj_matrix(shader.proj, pass->projection_matrix, &box);

	glUniform2f(shader.size, box.width, box.height);
	glUniform2f(shader.position, box.x, box.y);

	glUniform4fv(shader.colors, fx_options->gradient.count, (GLfloat*)fx_options->gradient.colors);
	glUniform1i(shader.count, fx_options->gradient.count);
	glUniform2f(shader.grad_size, fx_options->gradient.range.width, fx_options->gradient.range.height);
	glUniform1f(shader.degree, fx_options->gradient.degree);
	glUniform1f(shader.linear, fx_options->gradient.linear);
	glUniform1f(shader.blend, fx_options->gradient.blend);
	glUniform2f(shader.grad_box, fx_options->gradient.range.x, fx_options->gradient.range.y);
	glUniform2f(shader.origin, fx_options->gradient.origin[0], fx_options->gradient.origin[1]);

	struct fx_corner_fradii corners = fx_options->corners;
	uniform_corner_radii_set(&shader.radius, &corners);

	render(&box, options->clip, shader.pos_attrib);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

void fx_render_pass_add_box_shadow(struct fx_gles_render_pass *pass,
		const struct fx_render_box_shadow_options *options) {
	struct fx_renderer *renderer = pass->buffer->renderer;

	struct wlr_box box = options->box;
	assert(box.width > 0 && box.height > 0);
	if (options->blur_sigma <= 0.0f) {
		fx_render_pass_add_rounded_rect(pass,
				&(struct fx_render_rounded_rect_options){
			.base = {
				.box = box,
				.color = options->color,
				.clip = options->clip,
			},
			.corners = {
				.top_left = options->corner_radius,
				.top_right = options->corner_radius,
				.bottom_right = options->corner_radius,
				.bottom_left = options->corner_radius,
			},
			.clipped_region = options->clipped_region,
		});
		return;
	}

	pixman_region32_t clip_region;
	if (options->clip) {
		pixman_region32_init(&clip_region);
		pixman_region32_copy(&clip_region, options->clip);
	} else {
		pixman_region32_init_rect(&clip_region, box.x, box.y, box.width, box.height);
	}
	const struct wlr_box clipped_region_box = options->clipped_region.area;
	struct fx_corner_fradii clipped_region_corners = options->clipped_region.corners;
	apply_clip_region(&clip_region, &clipped_region_box, &clipped_region_corners);

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("Box (WxH, X, Y): %dx%d, %d, %d", box.width, box.height, box.x, box.y);
	TRACY_ZONE_TEXT_f("Clip Box (WxH, X, Y): %dx%d, %d, %d",
			clipped_region_box.width, clipped_region_box.height,
			clipped_region_box.x, clipped_region_box.y);
	TRACY_ZONE_TEXT_f("Clip Box Corners (TL, TR, BL, BR): %f, %f, %f, %f",
			clipped_region_corners.top_left,
			clipped_region_corners.top_right,
			clipped_region_corners.bottom_left,
			clipped_region_corners.bottom_right);
	TRACY_ZONE_TEXT_f("Shadow Options:");
	TRACY_ZONE_TEXT_f("\tColor RGBA: %f, %f, %f, %f",
			options->color.r, options->color.g, options->color.b, options->color.a);
	TRACY_ZONE_TEXT_f("\tBlur Sigma: %f", options->blur_sigma);
	push_fx_debug(renderer);

	// Blurred edges require blending, so just enable it
	setup_blending(WLR_RENDER_BLEND_MODE_PREMULTIPLIED);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(renderer->shaders.box_shadow.program);

	const struct wlr_render_color *color = &options->color;
	set_proj_matrix(renderer->shaders.box_shadow.proj, pass->projection_matrix, &box);
	glUniform4f(renderer->shaders.box_shadow.color, color->r, color->g, color->b, color->a);
	glUniform1f(renderer->shaders.box_shadow.blur_sigma, options->blur_sigma);
	glUniform2f(renderer->shaders.box_shadow.size, box.width, box.height);
	glUniform2f(renderer->shaders.box_shadow.position, box.x, box.y);
	glUniform1f(renderer->shaders.box_shadow.corner_radius, options->corner_radius);

	uniform_corner_radii_set(&renderer->shaders.box_shadow.clip_radius, &clipped_region_corners);

	glUniform2f(renderer->shaders.box_shadow.clip_position, clipped_region_box.x, clipped_region_box.y);
	glUniform2f(renderer->shaders.box_shadow.clip_size, clipped_region_box.width, clipped_region_box.height);

	render(&box, &clip_region, renderer->shaders.box_shadow.pos_attrib);
	pixman_region32_fini(&clip_region);

	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

// Renders the blur for each damaged rect and swaps the buffer.
// src_level is the downscale level of the current buffer's content, which only
// occupies the top-left (1 / 2^src_level) part of the buffer.
static void render_blur_segments(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options, struct blur_shader* shader,
		int src_level) {
	struct fx_render_texture_options *tex_options = &fx_options->tex_options;
	struct wlr_render_texture_options *options = &tex_options->base;
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct blur_data *blur_data = fx_options->blur_data;

	TRACY_BOTH_ZONES_START(renderer);
	push_fx_debug(renderer);

	// Swap fbo
	if (fx_options->current_buffer == pass->fx_offscreen_buffers->effects_buffer) {
		fx_framebuffer_bind(pass->fx_offscreen_buffers->effects_buffer_swapped);
	} else {
		fx_framebuffer_bind(pass->fx_offscreen_buffers->effects_buffer);
	}

	options->texture = fx_texture_from_buffer(&renderer->wlr_renderer,
			fx_options->current_buffer->buffer);
	struct fx_texture *texture = fx_get_texture(options->texture);

	/*
	 * Render
	 */

	struct wlr_box dst_box;
	struct wlr_fbox src_fbox;
	wlr_render_texture_options_get_src_box(options, &src_fbox);
	wlr_render_texture_options_get_dst_box(options, &dst_box);
	src_fbox.x /= options->texture->width;
	src_fbox.y /= options->texture->height;
	src_fbox.width /= options->texture->width;
	src_fbox.height /= options->texture->height;

	glDisable(GL_BLEND);
	glDisable(GL_STENCIL_TEST);

	glUseProgram(shader->program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex);

	switch (options->filter_mode) {
	case WLR_SCALE_FILTER_BILINEAR:
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case WLR_SCALE_FILTER_NEAREST:
		abort();
	}

	glUniform1i(shader->tex, 0);
	glUniform1f(shader->radius, blur_data->radius);

	if (shader == &renderer->shaders.blur1) {
		glUniform2f(shader->halfpixel,
				0.5f / (options->texture->width / 2.0f),
				0.5f / (options->texture->height / 2.0f));
	} else {
		glUniform2f(shader->halfpixel,
				0.5f / (options->texture->width * 2.0f),
				0.5f / (options->texture->height * 2.0f));
	}

	// Clamp samples to the texel centers of the valid source area, so that the
	// screen edges get extended instead of blending with stale buffer contents.
	// Matches the rounding of wlr_region_scale, which rounds the damage outwards.
	float tex_width = options->texture->width;
	float tex_height = options->texture->height;
	float level_width = ceilf(tex_width / (1 << src_level));
	float level_height = ceilf(tex_height / (1 << src_level));
	glUniform2f(shader->uv_min, 0.5f / tex_width, 0.5f / tex_height);
	glUniform2f(shader->uv_max,
			(level_width - 0.5f) / tex_width,
			(level_height - 0.5f) / tex_height);

	set_proj_matrix(shader->proj, pass->projection_matrix, &dst_box);
	set_tex_matrix(shader->tex_proj, options->transform, &src_fbox);

	render(&dst_box, options->clip, shader->pos_attrib);

	glBindTexture(texture->target, 0);
	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;

	wlr_texture_destroy(options->texture);

	// Swap buffer. We don't want to draw to the same buffer
	if (fx_options->current_buffer != pass->fx_offscreen_buffers->effects_buffer) {
		fx_options->current_buffer = pass->fx_offscreen_buffers->effects_buffer;
	} else {
		fx_options->current_buffer = pass->fx_offscreen_buffers->effects_buffer_swapped;
	}
}

static void render_blur_effects(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options) {
	struct fx_render_texture_options *tex_options = &fx_options->tex_options;
	struct wlr_render_texture_options *options = &tex_options->base;
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct blur_data *blur_data = fx_options->blur_data;
	struct fx_texture *texture = fx_get_texture(options->texture);

	struct blur_effects_shader shader = renderer->shaders.blur_effects;

	struct wlr_box dst_box;
	struct wlr_fbox src_fbox;
	wlr_render_texture_options_get_src_box(options, &src_fbox);
	wlr_render_texture_options_get_dst_box(options, &dst_box);

	src_fbox.x /= options->texture->width;
	src_fbox.y /= options->texture->height;
	src_fbox.width /= options->texture->width;
	src_fbox.height /= options->texture->height;

	glDisable(GL_BLEND);
	glDisable(GL_STENCIL_TEST);

	TRACY_BOTH_ZONES_START(renderer);
	push_fx_debug(renderer);

	glUseProgram(shader.program);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex);

	switch (options->filter_mode) {
	case WLR_SCALE_FILTER_BILINEAR:
		glTexParameteri(texture->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(texture->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		break;
	case WLR_SCALE_FILTER_NEAREST:
		abort();
	}

	glUniform1i(shader.tex, 0);
	glUniform1f(shader.noise, blur_data->noise);
	glUniform1f(shader.brightness, blur_data->brightness);
	glUniform1f(shader.contrast, blur_data->contrast);
	glUniform1f(shader.saturation, blur_data->saturation);

	set_proj_matrix(shader.proj, pass->projection_matrix, &dst_box);
	set_tex_matrix(shader.tex_proj, options->transform, &src_fbox);

	render(&dst_box, options->clip, shader.pos_attrib);

	glBindTexture(texture->target, 0);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;

	wlr_texture_destroy(options->texture);
}

// Blurs the fx_options current_buffer content and returns the blurred framebuffer.
// Returns NULL when the blur parameters reach 0.
static struct fx_framebuffer *get_main_buffer_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options) {
	if (pass->fx_offscreen_buffers == NULL) {
		wlr_log(WLR_ERROR, "FX Pass offscreen buffers not initialized. Skipping getting blur...");
		return NULL;
	}

	struct fx_renderer *renderer = pass->buffer->renderer;
	struct wlr_box buffer_bounds = {
		0, 0,
		fx_options->current_buffer->buffer->width, fx_options->current_buffer->buffer->height
	};

	// We don't want to affect the reference blur_data
	struct blur_data blur_data = blur_data_apply_strength(fx_options->blur_data, fx_options->blur_strength);
	if (fx_options->blur_strength <= 0 || !is_scene_blur_enabled(&blur_data)) {
		return NULL;
	}
	fx_options->blur_data = &blur_data;
	fx_options->tex_options.base.transform = WL_OUTPUT_TRANSFORM_NORMAL;

	pixman_region32_t damage;
	pixman_region32_init(&damage);
	pixman_region32_copy(&damage, fx_options->tex_options.base.clip);

	wlr_region_expand(&damage, &damage, blur_data_calc_size(&blur_data));
	// Make sure that the region doesn't expand past the buffer bounds
	pixman_region32_intersect_rect(&damage, &damage,
			0, 0, buffer_bounds.width, buffer_bounds.height);

	// damage region will be scaled, make a temp
	pixman_region32_t scaled_damage;
	pixman_region32_init(&scaled_damage);

	fx_options->tex_options.base.src_box = (struct wlr_fbox) {
		0, 0,
		buffer_bounds.width, buffer_bounds.height,
	};
	fx_options->tex_options.base.dst_box = buffer_bounds;
	// Clip the blur to the damage
	fx_options->tex_options.base.clip = &scaled_damage;
	// Artifacts with NEAREST filter
	fx_options->tex_options.base.filter_mode = WLR_SCALE_FILTER_BILINEAR;

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("dst_box (WxH, X, Y): %dx%d, %d, %d",
			fx_options->tex_options.base.dst_box.width,
			fx_options->tex_options.base.dst_box.height,
			fx_options->tex_options.base.dst_box.x,
			fx_options->tex_options.base.dst_box.y);
	TRACY_ZONE_TEXT_f("clip_box (WxH, X, Y): %dx%d, %d, %d",
			fx_options->tex_options.clip_box->width,
			fx_options->tex_options.clip_box->height,
			fx_options->tex_options.clip_box->x,
			fx_options->tex_options.clip_box->y);
	TRACY_ZONE_TEXT_f("Corners (TL, TR, BL, BR): %f, %f, %f, %f",
			fx_options->corners.top_left,
			fx_options->corners.top_right,
			fx_options->corners.bottom_left,
			fx_options->corners.bottom_right);
	TRACY_ZONE_TEXT_f("src_box (WxH, X, Y): %lfx%lf, %lf, %lf",
			fx_options->tex_options.base.src_box.width,
			fx_options->tex_options.base.src_box.height,
			fx_options->tex_options.base.src_box.x,
			fx_options->tex_options.base.src_box.y);
	TRACY_ZONE_TEXT_f("Ignore Transparent: %d", fx_options->ignore_transparent);
	TRACY_ZONE_TEXT_f("Discard Transparent: %d", fx_options->tex_options.discard_transparent);
	TRACY_ZONE_TEXT_f("Use Optimized Blur: %d", fx_options->use_optimized_blur);
	TRACY_ZONE_TEXT_f("Blur Options:");
	TRACY_ZONE_TEXT_f("\tNum Blur Passes: %d", fx_options->blur_data->num_passes);
	TRACY_ZONE_TEXT_f("\tBlur Radius: %f", fx_options->blur_data->radius);
	TRACY_ZONE_TEXT_f("\tBrightness: %f", fx_options->blur_data->brightness);
	TRACY_ZONE_TEXT_f("\tContrast: %f", fx_options->blur_data->contrast);
	TRACY_ZONE_TEXT_f("\tNoise: %f", fx_options->blur_data->noise);
	TRACY_ZONE_TEXT_f("\tSaturation: %f", fx_options->blur_data->saturation);
	push_fx_debug(renderer);

	// Downscale
	for (int i = 0; i < blur_data.num_passes; ++i) {
		wlr_region_scale(&scaled_damage, &damage, 1.0f / (1 << (i + 1)));
		render_blur_segments(pass, fx_options, &renderer->shaders.blur1, i);
	}

	// Upscale
	for (int i = blur_data.num_passes - 1; i >= 0; --i) {
		// when upsampling we make the region twice as big
		wlr_region_scale(&scaled_damage, &damage, 1.0f / (1 << i));
		render_blur_segments(pass, fx_options, &renderer->shaders.blur2, i + 1);
	}

	pixman_region32_fini(&scaled_damage);

	// Render additional blur effects like saturation, noise, contrast, etc...
	if (blur_data_should_parameters_blur_effects(&blur_data)
			&& pixman_region32_not_empty(&damage)) {
		if (fx_options->current_buffer == pass->fx_offscreen_buffers->effects_buffer) {
			fx_framebuffer_bind(pass->fx_offscreen_buffers->effects_buffer_swapped);
		} else {
			fx_framebuffer_bind(pass->fx_offscreen_buffers->effects_buffer);
		}
		fx_options->tex_options.base.clip = &damage;
		fx_options->tex_options.base.texture = fx_texture_from_buffer(
				&renderer->wlr_renderer, fx_options->current_buffer->buffer);
		render_blur_effects(pass, fx_options);
		if (fx_options->current_buffer != pass->fx_offscreen_buffers->effects_buffer) {
			fx_options->current_buffer = pass->fx_offscreen_buffers->effects_buffer;
		} else {
			fx_options->current_buffer = pass->fx_offscreen_buffers->effects_buffer_swapped;
		}
	}

	pixman_region32_fini(&damage);

	// Bind back to the default buffer
	fx_framebuffer_bind(pass->buffer);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;

	return fx_options->current_buffer;
}

// Liquid Glass's shape field (glass_field.frag): the panel's coverage
// blurred across into one buffer, then down into the other, over the glass
// node's box (`node`, which the shadow reaches past the panel). NULL when
// the buffers can't be had: the glass is then drawn without it.
static struct fx_framebuffer *render_glass_field(struct fx_gles_render_pass *pass,
		struct fx_texture *mask_tex, struct wlr_box mask_box, const float mask_src[4],
		float radius, struct wlr_box node, float sigma, int width, int height) {
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct fx_offscreen_buffers *fbos = pass->fx_offscreen_buffers;
	struct glass_field_shader *shader = &renderer->shaders.glass_field;
	bool failed = false;
	fx_framebuffer_get_or_create_half_float(renderer, width, height, &fbos->glass_field_buffer, &failed);
	fx_framebuffer_get_or_create_half_float(renderer, width, height, &fbos->glass_field_buffer_swapped, &failed);
	if (failed) {
		return NULL;
	}

	struct wlr_box dst_box = { .width = width, .height = height };
	struct wlr_fbox src_fbox = { .width = 1.0, .height = 1.0 };
	const int reach = (int)ceilf(3.0f * sigma) + 2;
	glDisable(GL_BLEND);
	glUseProgram(shader->program);
	set_proj_matrix(shader->proj, pass->projection_matrix, &dst_box);
	set_tex_matrix(shader->tex_proj, WL_OUTPUT_TRANSFORM_NORMAL, &src_fbox);
	if (mask_tex) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, mask_tex->tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glUniform1i(shader->mask, 1);
		glUniform4f(shader->mask_src, mask_src[0], mask_src[1], mask_src[2], mask_src[3]);
		glActiveTexture(GL_TEXTURE0);
	}
	glUniform1i(shader->has_mask, mask_tex != NULL);
	glUniform2f(shader->box_pos, mask_box.x, mask_box.y);
	glUniform2f(shader->box_size, mask_box.width, mask_box.height);
	glUniform1f(shader->radius, radius);
	glUniform1f(shader->sigma, sigma);
	glUniform2f(shader->texel, 1.0f / width, 1.0f / height);
	glUniform1i(shader->tex, 0);

	// Across, over rows reaching past the node by the kernel, which the
	// second pass reads.
	pixman_region32_t region;
	pixman_region32_init_rect(&region, node.x, node.y - reach, node.width, node.height + 2 * reach);
	fx_framebuffer_bind(fbos->glass_field_buffer_swapped);
	glUniform1i(shader->first, 1);
	glUniform2f(shader->dir, 1.0f, 0.0f);
	render(&dst_box, &region, shader->pos_attrib);
	pixman_region32_fini(&region);

	// Down, over the node (and a pixel round it, for glass.frag's gradient).
	pixman_region32_init_rect(&region, node.x - 1, node.y - 1, node.width + 2, node.height + 2);
	fx_framebuffer_bind(fbos->glass_field_buffer);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fbos->glass_field_buffer_swapped->tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glUniform1i(shader->first, 0);
	glUniform2f(shader->dir, 0.0f, 1.0f);
	render(&dst_box, &region, shader->pos_attrib);
	pixman_region32_fini(&region);
	glBindTexture(GL_TEXTURE_2D, 0);

	fx_framebuffer_bind(pass->buffer);
	glEnable(GL_BLEND);
	return fbos->glass_field_buffer;
}

// The blurred background, bent at a glass panel's edge (glass.frag).
static void render_glass(struct fx_gles_render_pass *pass, struct fx_texture *blur,
		const struct fx_render_blur_pass_options *fx_options, struct wlr_texture *mask,
		struct wlr_box mask_box, struct wlr_fbox mask_src) {
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct glass_shader *shader = &renderer->shaders.glass;
	const struct wlr_render_texture_options *options = &fx_options->tex_options.base;
	const int width = blur->wlr_texture.width, height = blur->wlr_texture.height;
	struct wlr_box dst_box = { .width = width, .height = height };
	struct wlr_fbox src_fbox = { .width = 1.0, .height = 1.0 };

	struct fx_texture *mask_tex = mask ? fx_get_texture(mask) : NULL;
	if (mask_tex && mask_tex->target != GL_TEXTURE_2D) {
		mask_tex = NULL;
	}
	if (!mask_tex) {
		mask_box = *fx_options->tex_options.clip_box;
	}
	const float mask_norm[4] = {
		mask ? mask_src.x / mask->width : 0, mask ? mask_src.y / mask->height : 0,
		mask ? mask_src.width / mask->width : 1, mask ? mask_src.height / mask->height : 1,
	};
	const float sigma = fmaxf(fx_options->refraction_thickness / 2.5f, 1.0f);
	// Shapes given: their exact geometry. Otherwise, the mask's blurred.
	const int shape_count = fx_options->glass_shape_count;
	struct fx_framebuffer *field = shape_count > 0 ? NULL : render_glass_field(pass, mask_tex, mask_box,
		mask_norm, fx_options->tex_options.corners.top_left, *fx_options->tex_options.clip_box, sigma, width, height);

	setup_blending(WLR_RENDER_BLEND_MODE_PREMULTIPLIED);

	pixman_region32_t clip_region;
	if (options->clip) {
		pixman_region32_init(&clip_region);
		pixman_region32_copy(&clip_region, options->clip);
	} else {
		pixman_region32_init_rect(&clip_region, 0, 0, width, height);
	}

	glUseProgram(shader->program);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(blur->target, blur->tex);
	glTexParameteri(blur->target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(blur->target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glUniform1i(shader->tex, 0);

	if (mask_tex) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, mask_tex->tex);
		// Linear, so the edge is found between pixels, not on them.
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glUniform1i(shader->mask, 1);
		glUniform4f(shader->mask_src, mask_norm[0], mask_norm[1], mask_norm[2], mask_norm[3]);
		glActiveTexture(GL_TEXTURE0);
	}
	glUniform1i(shader->has_mask, mask_tex != NULL);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, field ? field->tex : 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glUniform1i(shader->field, 2);
	glUniform2f(shader->field_texel, 1.0f / width, 1.0f / height);
	glUniform1f(shader->field_sigma, sigma);
	glActiveTexture(GL_TEXTURE0);
	GLfloat boxes[16 * 4], extras[16 * 2], clips[16 * 4];
	for (int i = 0; i < shape_count; i++) {
		const float *s = fx_options->glass_shapes[i];
		boxes[i * 4 + 0] = s[0];
		boxes[i * 4 + 1] = s[1];
		boxes[i * 4 + 2] = s[2];
		boxes[i * 4 + 3] = s[3];
		extras[i * 2 + 0] = s[4];
		extras[i * 2 + 1] = s[5];
		for (int k = 0; k < 4; k++)
			clips[i * 4 + k] = s[6 + k];
	}
	if (shape_count > 0) {
		glUniform4fv(shader->shape_box, shape_count, boxes);
		glUniform2fv(shader->shape_extra, shape_count, extras);
		glUniform4fv(shader->shape_clip, shape_count, clips);
	}
	glUniform1i(shader->shape_count, shape_count);

	glUniform2f(shader->texel, 1.0f / width, 1.0f / height);
	glUniform2f(shader->box_pos, mask_box.x, mask_box.y);
	glUniform2f(shader->box_size, mask_box.width, mask_box.height);
	glUniform1f(shader->radius, fx_options->tex_options.corners.top_left);
	glUniform1f(shader->refraction, fx_options->refraction);
	glUniform1f(shader->thickness, fx_options->refraction_thickness);
	glUniform1f(shader->alpha, wlr_render_texture_options_get_alpha(options));
	const struct wlr_scene_glass *g = fx_options->glass;
	const struct wlr_scene_glass plain = { .saturation = 1.0f, .light_dir = { 0.7071f, 0.7071f } };
	if (g == NULL) {
		g = &plain;
	}
	glUniform4f(shader->tint, g->tint[0], g->tint[1], g->tint[2], g->tint[3]);
	glUniform1f(shader->adapt, g->adapt);
	glUniform1f(shader->saturation, g->saturation);
	glUniform1f(shader->highlight, g->highlight);
	glUniform2f(shader->light_dir, g->light_dir[0], g->light_dir[1]);
	glUniform1f(shader->shadow, g->shadow);

	set_proj_matrix(shader->proj, pass->projection_matrix, &dst_box);
	set_tex_matrix(shader->tex_proj, WL_OUTPUT_TRANSFORM_NORMAL, &src_fbox);
	render(&dst_box, &clip_region, shader->pos_attrib);
	pixman_region32_fini(&clip_region);

	if (mask_tex) {
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(blur->target, 0);
}

void fx_render_pass_add_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options) {
	if (pass->fx_offscreen_buffers == NULL) {
		wlr_log(WLR_ERROR, "FX Pass offscreen buffers not initialized. Skipping blur...");
		return;
	}

	struct fx_renderer *renderer = pass->buffer->renderer;
	struct fx_render_texture_options *tex_options = &fx_options->tex_options;

	TRACY_BOTH_ZONES_START(renderer);
	push_fx_debug(renderer);

	const bool has_strength = fx_options->blur_strength < 1.0;
	struct fx_framebuffer *buffer = pass->fx_offscreen_buffers->optimized_blur_buffer;
	TRACY_ZONE_TEXT_f("Use Optimized Blur: %d", fx_options->use_optimized_blur);
	TRACY_ZONE_TEXT_f("Optimized Blur Successfully Used: %d",
			buffer && fx_options->use_optimized_blur);
	if (!fx_options->use_optimized_blur || has_strength) {
		// Render the blur into its own buffer
		struct fx_render_blur_pass_options blur_options = *fx_options;
		if (fx_options->use_optimized_blur && has_strength) {
			// Re-blur the saved non-blurred version of the optimized blur.
			// Isn't as efficient as just using the optimized blur buffer
			blur_options.current_buffer = pass->fx_offscreen_buffers->optimized_no_blur_buffer;
		} else {
			blur_options.current_buffer = pass->buffer;
		}
		buffer = get_main_buffer_blur(pass, &blur_options);
	}
	if (!buffer) {
		goto finish;
	}
	struct wlr_texture *wlr_texture =
		fx_texture_from_buffer(&renderer->wlr_renderer, buffer->buffer);
	struct fx_texture *blur_texture = fx_get_texture(wlr_texture);
	// The material's pattern sits where the blur is drawn (the window).
	const struct wlr_box pattern_box = tex_options->base.dst_box;

	// Get a stencil of the window ignoring transparent regions. Glass finds
	// its own shape in the mask (and draws its shadow outside it).
	const bool stencil = fx_options->ignore_transparent && fx_options->tex_options.base.texture &&
		fx_options->refraction <= 0;
	if (stencil) {
		stencil_mask_init();

		struct fx_render_texture_options tex_options = fx_options->tex_options;
		tex_options.discard_transparent = true;
		tex_options.clipped_region = fx_options->clipped_region;
		fx_render_pass_add_texture(pass, &tex_options);

		stencil_mask_close(true);
	}

	// Draw the blurred texture: bent at the edge for glass, flat otherwise.
	if (fx_options->refraction > 0) {
		render_glass(pass, blur_texture, fx_options, fx_options->tex_options.base.texture,
			fx_options->mask_box, fx_options->tex_options.base.src_box);
		wlr_texture_destroy(&blur_texture->wlr_texture);
		goto unstencil;
	}
	tex_options->base.dst_box = (struct wlr_box) {
		.x = 0,
		.y = 0,
		.width = buffer->buffer->width,
		.height = buffer->buffer->height,
	};
	tex_options->base.src_box = (struct wlr_fbox) {
		.x = 0,
		.y = 0,
		.width = buffer->buffer->width,
		.height = buffer->buffer->height,
	};
	tex_options->base.texture = &blur_texture->wlr_texture;
	// since we're capturing from the fbo, transform will always be normal
	tex_options->base.transform = WL_OUTPUT_TRANSFORM_NORMAL;
	tex_options->clipped_region = fx_options->clipped_region;
	tex_options->material = fx_options->blur_data ? fx_options->blur_data->material : 0;
	tex_options->material_box = pattern_box;
	fx_render_pass_add_texture(pass, tex_options);
	tex_options->material = 0;

	wlr_texture_destroy(&blur_texture->wlr_texture);

unstencil:
	// Finish stenciling
	if (stencil) {
		stencil_mask_fini();
	}

finish:
	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
}

bool fx_render_pass_add_optimized_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options) {
	if (pass->fx_offscreen_buffers == NULL) {
		wlr_log(WLR_ERROR, "FX Pass offscreen buffers not initialized. Skipping optimized blur...");
		return false;
	}
	struct fx_renderer *renderer = pass->buffer->renderer;
	struct wlr_box dst_box = fx_options->tex_options.base.dst_box;

	TRACY_BOTH_ZONES_START(renderer);
	TRACY_ZONE_TEXT_f("dst_box (WxH, X, Y): %dx%d, %d, %d",
			fx_options->tex_options.base.dst_box.width,
			fx_options->tex_options.base.dst_box.height,
			fx_options->tex_options.base.dst_box.x,
			fx_options->tex_options.base.dst_box.y);
	TRACY_ZONE_TEXT_f("clip_box (WxH, X, Y): %dx%d, %d, %d",
			fx_options->tex_options.clip_box->width,
			fx_options->tex_options.clip_box->height,
			fx_options->tex_options.clip_box->x,
			fx_options->tex_options.clip_box->y);
	TRACY_ZONE_TEXT_f("src_box (WxH, X, Y): %lfx%lf, %lf, %lf",
			fx_options->tex_options.base.src_box.width,
			fx_options->tex_options.base.src_box.height,
			fx_options->tex_options.base.src_box.x,
			fx_options->tex_options.base.src_box.y);
	TRACY_ZONE_TEXT_f("Ignore Transparent: %d", fx_options->ignore_transparent);
	TRACY_ZONE_TEXT_f("Discard Transparent: %d", fx_options->tex_options.discard_transparent);
	TRACY_ZONE_TEXT_f("Use Optimized Blur: %d", fx_options->use_optimized_blur);
	TRACY_ZONE_TEXT_f("Blur Options:");
	TRACY_ZONE_TEXT_f("\tNum Blur Passes: %d", fx_options->blur_data->num_passes);
	TRACY_ZONE_TEXT_f("\tBlur Radius: %f", fx_options->blur_data->radius);
	TRACY_ZONE_TEXT_f("\tBrightness: %f", fx_options->blur_data->brightness);
	TRACY_ZONE_TEXT_f("\tContrast: %f", fx_options->blur_data->contrast);
	TRACY_ZONE_TEXT_f("\tNoise: %f", fx_options->blur_data->noise);
	TRACY_ZONE_TEXT_f("\tSaturation: %f", fx_options->blur_data->saturation);
	push_fx_debug(renderer);

	pixman_region32_t clip;
	pixman_region32_init_rect(&clip,
			dst_box.x, dst_box.y, dst_box.width, dst_box.height);

	// Render the blur into its own buffer
	struct fx_render_blur_pass_options blur_options = *fx_options;
	blur_options.current_buffer = pass->buffer;
	blur_options.tex_options.base.clip = &clip;
	struct fx_framebuffer *fx_buffer = get_main_buffer_blur(pass, &blur_options);
	if (fx_buffer != NULL) {
		// Render the newly blurred content into the blur_buffer
		fx_render_pass_read_to_buffer(pass, &clip,
				pass->fx_offscreen_buffers->optimized_blur_buffer, fx_buffer);

		// Save the current scene pass state
		fx_render_pass_read_to_buffer(pass, &clip,
				pass->fx_offscreen_buffers->optimized_no_blur_buffer, pass->buffer);
	}

	pixman_region32_fini(&clip);

	pop_fx_debug(renderer);
	TRACY_BOTH_ZONES_END;
	return fx_buffer != NULL;
}

void fx_render_pass_read_to_buffer(struct fx_gles_render_pass *pass,
		pixman_region32_t *_region, struct fx_framebuffer *dst_buffer,
		struct fx_framebuffer *src_buffer) {
	if (!_region || !pixman_region32_not_empty(_region)) {
		return;
	}
	TRACY_BOTH_ZONES_START(pass->buffer->renderer);

	pixman_region32_t region;
	pixman_region32_init(&region);
	pixman_region32_copy(&region, _region);

	struct wlr_texture *src_tex =
		fx_texture_from_buffer(&pass->buffer->renderer->wlr_renderer, src_buffer->buffer);
	if (src_tex == NULL) {
		goto done;
	}

	// Draw onto the dst_buffer
	fx_framebuffer_bind(dst_buffer);
	wlr_render_pass_add_texture(&pass->base, &(struct wlr_render_texture_options) {
		.texture = src_tex,
		.clip = &region,
		.transform = WL_OUTPUT_TRANSFORM_NORMAL,
		.blend_mode = WLR_RENDER_BLEND_MODE_NONE,
		.dst_box = (struct wlr_box){
			.x = 0,
			.y = 0,
			.width = dst_buffer->buffer->width,
			.height = dst_buffer->buffer->height,
		},
		.src_box = (struct wlr_fbox){
			.x = 0,
			.y = 0,
			.width = src_buffer->buffer->width,
			.height = src_buffer->buffer->height,
		},
	});
	wlr_texture_destroy(src_tex);

	// Bind back to the main WLR buffer
	fx_framebuffer_bind(pass->buffer);

done:
	TRACY_BOTH_ZONES_END;

	pixman_region32_fini(&region);
}

static const char *reset_status_str(GLenum status) {
	switch (status) {
	case GL_GUILTY_CONTEXT_RESET_KHR:
		return "guilty";
	case GL_INNOCENT_CONTEXT_RESET_KHR:
		return "innocent";
	case GL_UNKNOWN_CONTEXT_RESET_KHR:
		return "unknown";
	default:
		return "<invalid>";
	}
}

struct fx_gles_render_pass *fx_begin_buffer_pass(struct fx_framebuffer *buffer,
		struct wlr_egl_context *prev_ctx, struct fx_render_timer *timer,
		struct wlr_drm_syncobj_timeline *signal_timeline, uint64_t signal_point) {
	struct fx_renderer *renderer = buffer->renderer;
	struct wlr_buffer *wlr_buffer = buffer->buffer;

	if (renderer->procs.glGetGraphicsResetStatusKHR) {
		GLenum status = renderer->procs.glGetGraphicsResetStatusKHR();
		if (status != GL_NO_ERROR) {
			wlr_log(WLR_ERROR, "GPU reset (%s)", reset_status_str(status));
			wl_signal_emit_mutable(&renderer->wlr_renderer.events.lost, NULL);
			return NULL;
		}
	}

	GLint fbo = fx_framebuffer_get_fbo(buffer);
	if (!fbo) {
		return NULL;
	}

	struct fx_gles_render_pass *pass = calloc(1, sizeof(*pass));
	if (pass == NULL) {
		return NULL;
	}

	wlr_render_pass_init(&pass->base, &render_pass_impl);
	wlr_buffer_lock(wlr_buffer);
	buffer->encoded_tf = 0; // until an HDR output pass says otherwise
	pass->buffer = buffer;
	pass->timer = timer;
	pass->prev_ctx = *prev_ctx;
	if (signal_timeline != NULL) {
		pass->signal_timeline = wlr_drm_syncobj_timeline_ref(signal_timeline);
		pass->signal_point = signal_point;
	}

	pass->fx_offscreen_buffers = NULL;
	pixman_region32_init(&pass->blur_padding_region);
	pass->has_blur = false;

	matrix_projection(pass->projection_matrix, wlr_buffer->width, wlr_buffer->height,
		WL_OUTPUT_TRANSFORM_FLIPPED_180);

	push_fx_debug(renderer);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);

	glViewport(0, 0, wlr_buffer->width, wlr_buffer->height);
	glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
	glDisable(GL_SCISSOR_TEST);

	pop_fx_debug(renderer);
	return pass;
}
