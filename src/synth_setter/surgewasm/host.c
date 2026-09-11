/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This translation unit includes Surge XT's pinned demo host so the sh_* ABI
 * remains the upstream implementation.
 */

#include <stdint.h>

#undef sh_start

#define FXP_HEADER_SIZE 60U

typedef struct {
  const uint8_t *bytes;
  uint64_t length;
  uint64_t offset;
} ss_memory_stream;

static bool ss_is_activated = false;
static bool ss_is_started = false;

static uint32_t ss_read_u32_be(const uint8_t *bytes) {
  return ((uint32_t)bytes[0] << 24U) | ((uint32_t)bytes[1] << 16U) |
         ((uint32_t)bytes[2] << 8U) | (uint32_t)bytes[3];
}

static int64_t CLAP_ABI ss_stream_read(const clap_istream_t *stream,
                                       void *buffer, uint64_t size) {
  ss_memory_stream *memory = stream->ctx;
  uint64_t remaining = memory->length - memory->offset;
  uint64_t count = size < remaining ? size : remaining;
  if (count == 0) return 0;
  memcpy(buffer, memory->bytes + memory->offset, count);
  memory->offset += count;
  return (int64_t)count;
}

EMSCRIPTEN_KEEPALIVE
int sh_start(double sample_rate, int max_frames) {
  if (!plugin->activate(plugin, sample_rate, 32, (uint32_t)max_frames))
    return 0;
  ss_is_activated = true;
  if (!plugin->start_processing(plugin)) return 0;
  ss_is_started = true;
  return 1;
}

EMSCRIPTEN_KEEPALIVE
int ss_load_fxp(const uint8_t *bytes, uint32_t length) {
  if (!plugin || !bytes || length < FXP_HEADER_SIZE + 4U) return 0;
  if (memcmp(bytes, "CcnK", 4) != 0 || memcmp(bytes + 8, "FPCh", 4) != 0)
    return 0;

  uint32_t payload_length = ss_read_u32_be(bytes + 56);
  if (payload_length != length - FXP_HEADER_SIZE ||
      memcmp(bytes + FXP_HEADER_SIZE, "sub3", 4) != 0)
    return 0;

  const clap_plugin_state_t *state =
      plugin->get_extension(plugin, CLAP_EXT_STATE);
  if (!state) return 0;

  ss_memory_stream memory = {bytes + FXP_HEADER_SIZE, payload_length, 0};
  clap_istream_t stream = {&memory, ss_stream_read};
  return state->load(plugin, &stream);
}

EMSCRIPTEN_KEEPALIVE
void ss_destroy(void) {
  if (plugin) {
    if (ss_is_started) {
      plugin->stop_processing(plugin);
      ss_is_started = false;
    }
    if (ss_is_activated) {
      plugin->deactivate(plugin);
      ss_is_activated = false;
    }
    plugin->destroy(plugin);
    plugin = NULL;
  }
  params = NULL;
  evq_n = 0;
  if (entry) {
    entry->deinit();
    entry = NULL;
  }
}
