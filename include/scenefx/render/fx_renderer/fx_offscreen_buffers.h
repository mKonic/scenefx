#ifndef _FX_OFFSCREEN_BUFFERS_H
#define _FX_OFFSCREEN_BUFFERS_H

#include <wlr/types/wlr_output.h>
#include <wlr/util/addon.h>

/**
 * Used to add effect framebuffers per output instead of every output sharing
 * them.
 */
struct fx_offscreen_buffers {
	struct wl_list link; // fx_renderer.offscreen_buffers
	struct wlr_addon addon;

	// Contains the blurred background for tiled windows
	struct fx_framebuffer *optimized_blur_buffer;
	// Contains the non-blurred background for tiled windows. Used for blurring
	// optimized surfaces with an alpha. Just as inefficient as the regular blur.
	struct fx_framebuffer *optimized_no_blur_buffer;
	// Contains the original pixels to draw over the areas where artifact are visible
	struct fx_framebuffer *blur_saved_pixels_buffer;
	// Blur swaps between the two effects buffers every time it scales the image
	// Buffer used for effects
	struct fx_framebuffer *effects_buffer;
	// Swap buffer used for effects
	struct fx_framebuffer *effects_buffer_swapped;
	// HDR (or any output color transform): the frame is drawn here in
	// half-float, then converted into the output's buffer.
	struct fx_framebuffer *blend_buffer;
	// With a screen shader: the frame after it, before the output pass.
	struct fx_framebuffer *screen_shader_buffer;
	// The display's correction table, as a texture (see output.frag), and
	// which version of it.
	struct wlr_texture *lut_texture;
	uint64_t lut_gen;
	// Liquid Glass's shape field, blurred across then down (glass_field.frag).
	struct fx_framebuffer *glass_field_buffer;
	struct fx_framebuffer *glass_field_buffer_swapped;
};

void fx_offscreen_buffers_destroy(struct fx_offscreen_buffers *fbos);
struct fx_offscreen_buffers *fx_offscreen_buffers_try_get(struct wlr_output *output);

#endif
