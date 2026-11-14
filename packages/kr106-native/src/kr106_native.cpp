#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <string_view>

#include "KR106_DSP.h"
#include "KR106_Presets_JUCE.h"

namespace py = pybind11;

namespace {
constexpr char kVersion[] = "2.5.13";
constexpr char kRevision[] = "bc15caee5843ab238a25d0969e68d57db2b1615f";
constexpr int kDspParameterCount = 56;
constexpr int kProgramParameter = 56;
constexpr int kBypassParameter = 57;
constexpr int kFadeSamples = 64;
constexpr float kHostBpm = 120.0F;

struct Parameter {
  int id;
  const char* name;
  float minimum;
  float maximum;
  const char* kind;
  float default_value;
};

constexpr std::array<Parameter, 58> kParameters{{
    {0, "Bender DCO", 0, 1, "float", 0},
    {1, "Bender VCF", 0, 1, "float", 0},
    {2, "Arp Rate", 0, 1, "float", 30.F / 128.F},
    {3, "LFO Rate", 0, 1, "float", .24F},
    {4, "LFO Delay", 0, 1, "float", 0},
    {5, "DCO LFO", 0, 1, "float", 0},
    {6, "DCO PWM", 0, 1, "float", .5F},
    {7, "DCO Sub", 0, 1, "float", 1},
    {8, "DCO Noise", 0, 1, "float", 0},
    {9, "HPF", 0, 3, "int", 1},
    {10, "VCF Freq", 0, 1, "float", 1},
    {11, "VCF Res", 0, 1, "float", 0},
    {12, "VCF Env", 0, 1, "float", 0},
    {13, "VCF LFO", 0, 1, "float", 0},
    {14, "VCF Kbd", 0, 1, "float", 0},
    {15, "Volume", 0, 1, "float", .5F},
    {16, "Attack", 0, 1, "float", .25F},
    {17, "Decay", 0, 1, "float", .25F},
    {18, "Sustain", 0, 1, "float", .9F},
    {19, "Release", 0, 1, "float", .25F},
    {20, "Transpose", 0, 1, "bool", 0},
    {21, "Hold", 0, 1, "bool", 0},
    {22, "Arpeggio", 0, 1, "bool", 0},
    {23, "Pulse", 0, 1, "bool", 1},
    {24, "Saw", 0, 1, "bool", 1},
    {25, "Sub Sw", 0, 1, "bool", 1},
    {26, "Chorus Off", 0, 1, "bool", 0},
    {27, "Chorus I", 0, 1, "bool", 1},
    {28, "Chorus II", 0, 1, "bool", 0},
    {29, "Octave", 0, 2, "int", 1},
    {30, "Arp Mode", 0, 2, "int", 0},
    {31, "Arp Range", 0, 2, "int", 0},
    {32, "LFO Mode", 0, 1, "int", 0},
    {33, "PWM Mode", 0, 2, "int", 1},
    {34, "VCF Env Inv", 0, 1, "int", 0},
    {35, "VCA Mode", 0, 1, "int", 1},
    {36, "Bender", -1, 1, "float", 0},
    {37, "Tuning", -1, 1, "float", 0},
    {38, "Power", 0, 1, "bool", 1},
    {39, "Porta Mode", 0, 2, "int", 1},
    {40, "Porta Rate", 0, 1, "float", 0},
    {41, "Transpose Offset", -24, 36, "int", 0},
    {42, "Bender LFO", 0, 1, "float", 0},
    {43, "ADSR Mode", 0, 1, "int", 1},
    {44, "Master Volume", 0, 1, "float", .5F},
    {45, "Voices", 6, 10, "int", 6},
    {46, "VCF Oversample", 1, 4, "int", 2},
    {47, "Ignore Velocity", 0, 1, "bool", 1},
    {48, "Arp Limit Kbd", 0, 1, "bool", 1},
    {49, "Arp Sync Host", 0, 1, "bool", 0},
    {50, "LFO Sync Host", 0, 1, "bool", 0},
    {51, "Mono Retrigger", 0, 1, "bool", 1},
    {52, "Send MIDI SysEx", 0, 1, "bool", 0},
    {53, "Arp Quantize", 0, 8, "int", 6},
    {54, "LFO Quantize", 0, 12, "int", 5},
    {55, "Oscillator Mode", 0, 1, "int", 1},
    {56, "Program", 0, 127, "int", 0},
    {57, "Bypass", 0, 1, "bool", 0},
}};

static_assert(kParameters[55].id == kDspParameterCount - 1);

bool is_unit_float(int id) {
  return std::string_view(kParameters[id].kind) == "float" &&
         kParameters[id].minimum == 0 && kParameters[id].maximum == 1;
}

void validate_parameter(int id, float value) {
  if (id < 0 || id >= static_cast<int>(kParameters.size())) {
    throw py::value_error("unknown parameter id");
  }
  const auto& parameter = kParameters[id];
  if (!std::isfinite(value) || value < parameter.minimum ||
      value > parameter.maximum) {
    throw py::value_error("parameter value is outside its native range");
  }
  if (std::string_view(parameter.kind) == "int" && std::floor(value) != value) {
    throw py::value_error("integer parameter requires an integer native value");
  }
  if (std::string_view(parameter.kind) == "bool" && value != 0.F &&
      value != 1.F) {
    throw py::value_error(
        "boolean parameter requires a native value of 0 or 1");
  }
}

bool is_live_performance_parameter(int id) {
  switch (id) {
    case 0:
    case 1:
    case 2:
    case 20:
    case 21:
    case 22:
    case 30:
    case 31:
    case 36:
    case 37:
    case 38:
    case 39:
    case 40:
    case 41:
    case 42:
    case 43:
    case 44:
    case 45:
    case 46:
    case 47:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 55:
      return true;
    default:
      return false;
  }
}

void apply_parameter(KR106DSP<float>* dsp, int id, float value) {
  switch (id) {
    case 38:
      if (value <= .5F) dsp->PowerOff();
      return;
    case 41:
      dsp->SetKeyTranspose(static_cast<int>(value));
      return;
    case 44:
      dsp->mMasterVol = value * value;
      dsp->mMasterVolSmooth = dsp->mMasterVol;
      return;
    case 45: {
      int voices = static_cast<int>(value + .5F);
      voices = voices <= 7 ? 6 : voices <= 9 ? 8 : 10;
      dsp->SetActiveVoices(voices);
      return;
    }
    case 46:
      dsp->SetOversample(value <= 1 ? 1 : value <= 3 ? 2 : 4);
      return;
    case 47:
      dsp->mIgnoreVelocity = value > .5F;
      return;
    case 48:
      dsp->mArp.mLimitToKeyboard = value > .5F;
      return;
    case 49:
    case 50:
      return;
    case 52:  // MIDI-out-only; deliberately has no DSP audio dispatch.
      return;
    case 51:
      dsp->mMonoRetrigger = value > .5F;
      return;
    case 55:
      dsp->ForEachVoice([value](kr106::Voice<float>& voice) {
        voice.mOscMode = static_cast<int>(value + .5F);
      });
      return;
    default:
      dsp->SetParam(id, value);
  }
}

void apply_defaults(KR106DSP<float>* dsp) {
  apply_parameter(dsp, 43, kParameters[43].default_value);
  for (int id = 0; id < kDspParameterCount; ++id) {
    if (id != 43) apply_parameter(dsp, id, kParameters[id].default_value);
  }
}

void load_program(KR106DSP<float>* dsp, int program,
                  std::array<float, 58>* effective_values) {
  const bool j106_model = (*effective_values)[43] > .5F;
  const auto& preset = kFactoryPresets[program + (j106_model ? 128 : 0)];
  for (int id = 0; id < 44; ++id) {
    if (is_live_performance_parameter(id)) continue;
    float value = static_cast<float>(preset.values[id]);
    if (is_unit_float(id)) value /= 127.F;
    if (id == 5 && j106_model)
      value = kr106::Voice<float>::dcoLfoDepth106_inverseTaper(value);
    if (id == 6 && j106_model) value = std::min(1.F, value * (127.F / 105.F));
    (*effective_values)[id] = value;
    apply_parameter(dsp, id, value);
  }
}

struct ParameterValues {
  std::array<bool, 58> is_explicit{};
  std::array<float, 58> values{};
};

ParameterValues collect_parameters(const py::dict& parameters,
                                   int maximum_id = kBypassParameter) {
  ParameterValues result;
  for (const auto& parameter : kParameters)
    result.values[parameter.id] = parameter.default_value;
  for (const auto& item : parameters) {
    const int id = py::cast<int>(item.first);
    const float value = py::cast<float>(item.second);
    validate_parameter(id, value);
    if (id > maximum_id)
      throw py::value_error("parameter is not accepted in this argument");
    result.is_explicit[id] = true;
    result.values[id] = value;
  }
  return result;
}

py::array_t<float> render_note(const py::dict& parameters, int midi_note,
                               int velocity, int start_sample, int end_sample,
                               int num_samples, float sample_rate,
                               int block_size,
                               const py::dict& baseline_parameters) {
  if (midi_note < 0 || midi_note > 127 || velocity < 0 || velocity > 127) {
    throw py::value_error("MIDI note and velocity must be in [0, 127]");
  }
  if (num_samples <= 0 || start_sample < 0 || end_sample <= start_sample ||
      end_sample > num_samples) {
    throw py::value_error(
        "sample interval must satisfy 0 <= start < end <= num_samples");
  }
  if (!std::isfinite(sample_rate) || sample_rate <= 0 || block_size <= 0) {
    throw py::value_error("sample_rate and block_size must be positive");
  }

  const auto parameters_to_apply = collect_parameters(parameters);
  const auto baseline_to_apply =
      collect_parameters(baseline_parameters, kDspParameterCount - 1);
  std::array<float, 58> effective_values{};
  for (const auto& parameter : kParameters)
    effective_values[parameter.id] = parameter.default_value;

  KR106DSP<float> dsp(KR106DSP<float>::kMaxVoices);
  dsp.Reset(sample_rate, block_size);
  apply_defaults(&dsp);
  for (int id = 0; id < kDspParameterCount; ++id) {
    if (!baseline_to_apply.is_explicit[id]) continue;
    effective_values[id] = baseline_to_apply.values[id];
    apply_parameter(&dsp, id, effective_values[id]);
  }
  if (parameters_to_apply.is_explicit[kProgramParameter]) {
    load_program(
        &dsp, static_cast<int>(parameters_to_apply.values[kProgramParameter]),
        &effective_values);
  }
  for (int id = 0; id < kDspParameterCount; ++id) {
    if (!parameters_to_apply.is_explicit[id]) continue;
    effective_values[id] = parameters_to_apply.values[id];
    apply_parameter(&dsp, id, effective_values[id]);
  }

  const bool bypassed = parameters_to_apply.values[kBypassParameter] > .5F;
  const bool transpose_mode = effective_values[20] > .5F;
  const bool arp_sync = effective_values[49] > .5F;
  const bool lfo_sync = effective_values[50] > .5F;
  auto result = py::array_t<float>({2, num_samples});
  float* audio = result.mutable_data();
  int position = 0;
  bool note_started = false;
  bool note_ended = false;

  while (position < num_samples) {
    if (!note_started && position == start_sample) {
      if (velocity == 0) {
        dsp.NoteOff(midi_note);
      } else if (transpose_mode) {
        apply_parameter(&dsp, 41, static_cast<float>(midi_note - 60));
      } else {
        dsp.NoteOn(midi_note, velocity);
      }
      note_started = true;
    }
    if (!note_ended && position == end_sample) {
      if (!transpose_mode) dsp.NoteOff(midi_note);
      note_ended = true;
    }

    int next_event = num_samples;
    if (!note_started)
      next_event = start_sample;
    else if (!note_ended)
      next_event = end_sample;
    const int frames = std::min(block_size, next_event - position);
    if (frames == 0) continue;

    dsp.mArp.mSyncToHost = arp_sync;
    dsp.mArp.mHostPlaying = arp_sync;
    dsp.mArp.mHostBPM = kHostBpm;
    dsp.mArp.mHostBeatPos = position * kHostBpm / (60.F * sample_rate);
    dsp.mLFO.mSyncToHost = lfo_sync;
    dsp.mLFO.mHostPlaying = true;
    dsp.mLFO.mHostBPM = kHostBpm;

    float* channels[] = {audio + position, audio + num_samples + position};
    dsp.ProcessBlock(nullptr, channels, 2, frames);
    for (int index = 0; index < frames; ++index) {
      const int sample = position + index;
      const float fade = sample < kFadeSamples
                             ? static_cast<float>(sample) / kFadeSamples
                             : 1.F;
      if (bypassed || effective_values[38] <= .5F) {
        channels[0][index] = 0.F;
        channels[1][index] = 0.F;
      } else {
        channels[0][index] *= fade;
        channels[1][index] *= fade;
      }
    }
    position += frames;
  }
  return result;
}
}  // namespace

PYBIND11_MODULE(kr106_native, module) {
  module.doc() =
      "Pinned KR-106 native offline renderer; calls retain the Python GIL.";
  module.def("get_version", [] { return kVersion; });
  module.def("get_source_revision", [] { return kRevision; });
  module.def("get_parameters", [] {
    py::list result;
    for (const auto& parameter : kParameters) {
      py::dict item;
      item["id"] = parameter.id;
      item["name"] = parameter.name;
      item["minimum"] = parameter.minimum;
      item["maximum"] = parameter.maximum;
      item["kind"] = parameter.kind;
      item["default"] = parameter.default_value;
      result.append(std::move(item));
    }
    return result;
  });
  module.def("render_note", &render_note, py::arg("parameters"),
             py::arg("midi_note"), py::arg("velocity"), py::arg("start_sample"),
             py::arg("end_sample"), py::arg("num_samples"),
             py::arg("sample_rate"), py::arg("block_size") = 512, py::kw_only(),
             py::arg("baseline_parameters") = py::dict());
}
