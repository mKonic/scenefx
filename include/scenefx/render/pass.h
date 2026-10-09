#ifndef SCENE_FX_RENDER_PASS_H
#define SCENE_FX_RENDER_PASS_H

#include <stdbool.h>
#include <wlr/render/pass.h>
#include <wlr/render/interface.h>
#include <wlr/render/swapchain.h>

#include "render/egl.h"
#include "types/fx/clipped_region.h"

struct wlr_scene_glass;

struct fx_gles_render_pass {
	struct wlr_render_pass base;
	struct fx_framebuffer *buffer;
	float projection_matrix[9];
	struct wlr_egl_context prev_ctx;
	struct fx_render_timer *timer;
	struct wlr_drm_syncobj_timeline *signal_timeline;
	uint64_t signal_point;

	// The region where there's blur
	pixman_region32_t blur_padding_region;
	bool has_blur;
	// Contains output-specific framebuffers.
	// NULL when no advanced effects like blur is being used in the current pass.
	// Call `fx_render_pass_init_offscreen_buffers` to use advanced effects.
	struct fx_offscreen_buffers *fx_offscreen_buffers;

	// The output's color transform needs a second pass (HDR): the frame is
	// drawn into the output's half-float blend buffer, then converted into
	// `output_buffer` with `output_matrix` and `output_tf` at submit.
	bool two_pass;
	struct fx_framebuffer *output_buffer;
	float output_matrix[9];
	int output_tf; // as output.frag's out_tf
	// The display's correction (its colour profile): a matrix in linear
	// light after output_matrix, and a 3D table on the encoded signal
	// (wlr_scene_output_set_correction). Either forces the second pass.
	bool has_calibration;
	float calibration[9];
	const float *lut;
	int lut_size;
	uint64_t lut_gen;
};

struct fx_gradient {
	float degree;
	/* The full area the gradient fit too, for borders use the window size */
	struct wlr_box range;
	/* The center of the gradient, {0.5, 0.5} for normal*/
	float origin[2];
	/* 1 = Linear, 2 = Conic */
	int linear;
	/* Whether or not to blend the colors */
	int blend;
	int count;
	float *colors;
};

struct fx_render_texture_options {
	struct wlr_render_texture_options base;
	const struct wlr_box *clip_box; // Used to clip csd. Ignored if NULL
	struct fx_corner_fradii corners;
	bool discard_transparent;
	// With discard_transparent: alpha at or below this counts as clear.
	float discard_below;
	struct clipped_fregion clipped_region;
	// Drawn bent over a grid of (warp_cols + 1) * (warp_rows + 1) points,
	// each x, y in the render buffer then u, v across dst_box (0 to 1),
	// instead of filling dst_box. warp_box is the unbent rectangle, which the
	// corners are rounded at. Clipped to base.clip.
	const float *warp;
	int warp_cols, warp_rows;
	struct wlr_box warp_box;
	// Motion blur, with more than one sample: the texture fills motion_box
	// (render buffer coordinates) and is averaged over that many copies back
	// along motion_back_x, motion_back_y, all over dst_box.
	int motion_samples;
	struct wlr_fbox motion_box;
	float motion_back_x, motion_back_y;
	// Saturation and brightness to draw it with; NULL as it is.
	const float *tint;
	// A blur's material (struct blur_data.material), its pattern at
	// material_box (render buffer coordinates).
	int material;
	struct wlr_box material_box;
};

struct fx_render_rect_options {
	struct wlr_render_rect_options base;
	struct clipped_fregion clipped_region;
};

struct fx_render_rect_grad_options {
	struct wlr_render_rect_options base;
	struct fx_gradient gradient;
};

struct fx_render_rounded_rect_options {
	struct wlr_render_rect_options base;
	struct fx_corner_fradii corners;
	struct clipped_fregion clipped_region;
};

struct fx_render_rounded_rect_grad_options {
	struct wlr_render_rect_options base;
	struct fx_gradient gradient;
	struct fx_corner_fradii corners;
};

struct fx_render_box_shadow_options {
	struct wlr_box box;
	struct clipped_fregion clipped_region;
	/* Clip region, leave NULL to disable clipping */
	const pixman_region32_t *clip;

	float blur_sigma;
	int corner_radius;
	struct wlr_render_color color;
};

struct fx_render_blur_pass_options {
	struct fx_render_texture_options tex_options;
	struct fx_framebuffer *current_buffer;
	struct blur_data *blur_data;
	bool use_optimized_blur;
	bool ignore_transparent;
	float blur_strength;
	struct fx_corner_fradii corners;
	struct clipped_fregion clipped_region;
	// Liquid Glass: bend the background by up to `refraction` pixels within
	// `refraction_thickness` of the edge. 0 draws it flat, as before.
	float refraction;
	float refraction_thickness;
	// The glass's material (when refraction > 0): see struct wlr_scene_glass.
	const struct wlr_scene_glass *glass;
	// Where the mask (the panel's buffer) is, in output buffer pixels: the
	// glass node may reach past it (for its shadow).
	struct wlr_box mask_box;
	// The glass's shapes, in output buffer pixels: x, y, width, height,
	// radius, opacity. None: its shape is the mask's.
	float glass_shapes[16][10];  // box, radius, opacity, clip box (width < 0: none)
	int glass_shape_count;
};

struct fx_gles_render_pass *fx_get_render_pass(struct wlr_render_pass *render_pass);

/**
 * The display's correction for this pass (call before
 * fx_render_pass_init_offscreen_buffers): a row-major matrix in linear light
 * (NULL: none) and a lut_size^3 RGB table on the encoded signal, red fastest
 * (NULL: none); lut_gen changes whenever the table does.
 */
void fx_render_pass_set_correction(struct wlr_render_pass *render_pass,
	const float *calibration, const float *lut, int lut_size, uint64_t lut_gen);

/**
 * Initializes the render pass offscreen buffers required for advanced effects
 * like blur.
 */
bool fx_render_pass_init_offscreen_buffers(struct wlr_render_pass *render_pass,
		struct wlr_output *output);

/**
 * Render a fx texture.
 */
void fx_render_pass_add_texture(struct fx_gles_render_pass *render_pass,
	const struct fx_render_texture_options *options);

/**
 * Render a rectangle.
 */
void fx_render_pass_add_rect(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rect_options *options);

/**
 * Render a rectangle with a gradient.
 */
void fx_render_pass_add_rect_grad(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rect_grad_options *options);

/**
 * Render a rounded rectangle.
 */
void fx_render_pass_add_rounded_rect(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rounded_rect_options *options);

/**
 * Render a rounded rectangle with a gradient.
 */
void fx_render_pass_add_rounded_rect_grad(struct fx_gles_render_pass *render_pass,
	const struct fx_render_rounded_rect_grad_options *options);

/**
 * Render a box shadow.
 */
void fx_render_pass_add_box_shadow(struct fx_gles_render_pass *pass,
		const struct fx_render_box_shadow_options *options);

/**
 * Render blur.
 */
void fx_render_pass_add_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options);

/**
 * Render optimized blur.
 */
bool fx_render_pass_add_optimized_blur(struct fx_gles_render_pass *pass,
		struct fx_render_blur_pass_options *fx_options);

/**
 * Render from one buffer to another
 */
void fx_render_pass_read_to_buffer(struct fx_gles_render_pass *pass,
		pixman_region32_t *region, struct fx_framebuffer *dst_buffer,
		struct fx_framebuffer *src_buffer);

#endif
