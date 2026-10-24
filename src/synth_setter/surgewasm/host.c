/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This translation unit includes Surge XT's pinned demo host so the sh_* ABI
 * remains the upstream implementation.
 */

#include <stdint.h>
#include <stdlib.h>

#undef sh_start

#define FXP_HEADER_SIZE 60U
#define SS_INITIAL_STATE_CAPACITY (64U * 1024U)
#define SS_MAX_STATE_SIZE (64U * 1024U * 1024U)

typedef struct {
  const uint8_t *bytes;
  uint64_t length;
  uint64_t offset;
} ss_memory_stream;

typedef struct {
  uint8_t *bytes;
  uint32_t length;
  uint32_t capacity;
  bool failed;
} ss_state_collector;

static bool ss_is_activated = false;
static bool ss_is_started = false;
static uint8_t *ss_saved_fxp = NULL;
static uint32_t ss_saved_fxp_length = 0;

static uint32_t ss_read_u32_be(const uint8_t *bytes) {
  return ((uint32_t)bytes[0] << 24U) | ((uint32_t)bytes[1] << 16U) |
         ((uint32_t)bytes[2] << 8U) | (uint32_t)bytes[3];
}

static void ss_write_u32_be(uint8_t *bytes, uint32_t value) {
  bytes[0] = (uint8_t)(value >> 24U);
  bytes[1] = (uint8_t)(value >> 16U);
  bytes[2] = (uint8_t)(value >> 8U);
  bytes[3] = (uint8_t)value;
}

static int64_t CLAP_ABI ss_stream_write(const clap_ostream_t *stream,
                                        const void *buffer, uint64_t size) {
  ss_state_collector *collector = stream->ctx;
  if (collector->failed) return -1;
  if (size > SS_MAX_STATE_SIZE - collector->length) {
    collector->failed = true;
    return -1;
  }

  uint32_t required = collector->length + (uint32_t)size;
  if (required > collector->capacity) {
    uint32_t capacity =
        collector->capacity ? collector->capacity : SS_INITIAL_STATE_CAPACITY;
    while (capacity < required) {
      capacity =
          capacity > SS_MAX_STATE_SIZE / 2U ? SS_MAX_STATE_SIZE : capacity * 2U;
    }
    uint8_t *bytes = realloc(collector->bytes, capacity);
    if (!bytes) {
      collector->failed = true;
      return -1;
    }
    collector->bytes = bytes;
    collector->capacity = capacity;
  }

  if (size > 0) memcpy(collector->bytes + collector->length, buffer, size);
  collector->length = required;
  return (int64_t)size;
}

static void ss_release_fxp(void) {
  free(ss_saved_fxp);
  ss_saved_fxp = NULL;
  ss_saved_fxp_length = 0;
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
int ss_save_fxp(void) {
  ss_release_fxp();
  if (!plugin) return 0;

  const clap_plugin_state_t *state =
      plugin->get_extension(plugin, CLAP_EXT_STATE);
  if (!state) return 0;

  ss_state_collector collector = {NULL, 0, 0, false};
  clap_ostream_t stream = {&collector, ss_stream_write};
  bool saved = state->save(plugin, &stream);
  if (!saved || collector.failed || collector.length < 4U ||
      memcmp(collector.bytes, "sub3", 4) != 0) {
    free(collector.bytes);
    return 0;
  }

  uint32_t fxp_length = FXP_HEADER_SIZE + collector.length;
  uint8_t *fxp = malloc(fxp_length);
  if (!fxp) {
    free(collector.bytes);
    return 0;
  }
  memset(fxp, 0, FXP_HEADER_SIZE);
  memcpy(fxp, "CcnK", 4);
  memcpy(fxp + 8, "FPCh", 4);
  ss_write_u32_be(fxp + 12, 1U);
  memcpy(fxp + 16, "cjs3", 4);
  ss_write_u32_be(fxp + 20, 1U);
  ss_write_u32_be(fxp + 24, 1U);
  ss_write_u32_be(fxp + 56, collector.length);
  memcpy(fxp + FXP_HEADER_SIZE, collector.bytes, collector.length);
  free(collector.bytes);

  ss_saved_fxp = fxp;
  ss_saved_fxp_length = fxp_length;
  return 1;
}

EMSCRIPTEN_KEEPALIVE
const uint8_t *ss_fxp_data(void) { return ss_saved_fxp; }

EMSCRIPTEN_KEEPALIVE
uint32_t ss_fxp_length(void) { return ss_saved_fxp_length; }

EMSCRIPTEN_KEEPALIVE
void ss_free_fxp(void) { ss_release_fxp(); }

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
  ss_release_fxp();
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
