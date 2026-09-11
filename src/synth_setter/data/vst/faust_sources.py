"""Checked-in Faust source strings keyed by parameter-spec identity.

Usage::

    import dawdreamer as dd

    dsp = resolve_faust_dsp(ParamSpecName("faust_bright_organ"))
    engine = dd.RenderEngine(44100, 128)
    processor = engine.make_faust_processor("faust")
    processor.num_voices = dsp.num_voices
    processor.set_dsp_string(dsp.source)
    processor.compile()
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from synth_setter.data.vst.faust_shimmer_fdn_source import SHIMMER_FDN_SOURCE
from synth_setter.param_spec_name import ParamSpecName


@dataclass(frozen=True)
class FaustDsp:
    """One in-memory Faust program and its compile-time geometry.

    .. attribute :: source

       Complete source passed to ``set_dsp_string``.

    .. attribute :: num_voices

       Faust polyphony count; zero selects monophonic compilation.

    .. attribute :: outputs

       Native output channel count.
    """

    source: str
    num_voices: int
    outputs: int


_BRIGHT_ORGAN_SOURCE = r"""import("stdfaust.lib");

declare name "brightOrgan";
declare author "Claude AI";

MAX_FREQ_MARGIN_HZ = 1000;
MIN_FILTER_FREQ_HZ = 20;
MAX_RESONANCE_RATIO = 0.4;
MAX_LOWPASS_RATIO = 0.9;
MAX_FREQ = ma.SR / 2 - MAX_FREQ_MARGIN_HZ;

safe_resonlp(f, q, gain) = fi.resonlp(min(f, MAX_FREQ * MAX_RESONANCE_RATIO), q, gain);
safe_lowpass(order, f) = fi.lowpass(order, max(MIN_FILTER_FREQ_HZ, min(f, MAX_FREQ * MAX_LOWPASS_RATIO)));

freq = hslider("h:Main/freq [style:knob][midi:ctrl 1]", 220, 55, 880, 0.1);
gate = button("h:Main/gate [midi:ctrl 64]");
volume = hslider("h:Main/volume [style:knob][midi:ctrl 7]", 0.5, 0, 1, 0.01);

foundation8 = hslider("h:Stops/Foundation 8' [style:knob][midi:ctrl 14]", 0.8, 0, 1, 0.01);
principal4 = hslider("h:Stops/Principal 4' [style:knob][midi:ctrl 15]", 0.5, 0, 1, 0.01);
fifteenth2 = hslider("h:Stops/Fifteenth 2' [style:knob][midi:ctrl 16]", 0.3, 0, 1, 0.01);
flute8 = hslider("h:Stops/Flute 8' [style:knob][midi:ctrl 17]", 0.4, 0, 1, 0.01);
nasard = hslider("h:Stops/Nasard 2 2/3' [style:knob][midi:ctrl 18]", 0.2, 0, 1, 0.01);
tierce = hslider("h:Stops/Tierce 1 3/5' [style:knob][midi:ctrl 19]", 0.15, 0, 1, 0.01);

reverbAmount = hslider("h:Reverb/Amount [style:knob][midi:ctrl 91]", 0.3, 0, 1, 0.01);
reverbDamp = hslider("h:Reverb/Damp [style:knob][midi:ctrl 92]", 0.5, 0, 1, 0.01);
reverbSize = hslider("h:Reverb/Size [style:knob][midi:ctrl 93]", 0.6, 0, 1, 0.01);

BRIGHTNESS_BASE = 2;
BRIGHTNESS_RANGE = 8;
ORGAN_BREATH_GAIN = 0.015;
FLUTE_BREATH_GAIN = 0.02;
FLUTE_SECOND_GAIN = 0.25;
FLUTE_THIRD_GAIN = 0.08;

organ_pipe(f, brightness) = pipe
with {
    safe_f = min(f, MAX_FREQ);
    square = os.square(safe_f);
    cutoff = safe_f * (BRIGHTNESS_BASE + brightness * BRIGHTNESS_RANGE);
    filtered = square : safe_lowpass(2, cutoff);
    breath = no.noise * ORGAN_BREATH_GAIN : safe_resonlp(safe_f, 1, 1);
    pipe = filtered + breath;
};

flute_pipe(f) = pipe
with {
    safe_f = min(f, MAX_FREQ);
    fundamental = os.osc(safe_f);
    h2 = os.osc(min(safe_f * 2, MAX_FREQ)) * FLUTE_SECOND_GAIN * (safe_f * 2 < MAX_FREQ);
    h3 = os.osc(min(safe_f * 3, MAX_FREQ)) * FLUTE_THIRD_GAIN * (safe_f * 3 < MAX_FREQ);
    breath = no.noise * FLUTE_BREATH_GAIN : safe_resonlp(safe_f * 1.5, 2, 1);
    pipe = fundamental + h2 + h3 + breath;
};

PRINCIPAL_FREQ_RATIO = 2;
FIFTEENTH_FREQ_RATIO = 4;
NASARD_FREQ_RATIO = 3;
TIERCE_FREQ_RATIO = 5;
NASARD_GAIN = 0.7;
TIERCE_GAIN = 0.5;
STOP_NORMALIZATION = 4;

organ = (
    organ_pipe(freq, 0.3) * foundation8 +
    organ_pipe(freq * PRINCIPAL_FREQ_RATIO, 0.4) * principal4 +
    organ_pipe(freq * FIFTEENTH_FREQ_RATIO, 0.5) * fifteenth2 +
    flute_pipe(freq) * flute8 +
    flute_pipe(freq * NASARD_FREQ_RATIO) * nasard * NASARD_GAIN +
    flute_pipe(freq * TIERCE_FREQ_RATIO) * tierce * TIERCE_GAIN
) / STOP_NORMALIZATION;

ATTACK_SECONDS = 0.1;
RELEASE_SECONDS = 0.2;
REVERB_SPREAD = 0.3;
REVERB_CUTOFF_HZ = 6000;
REVERB_DRY_SCALE = 0.5;

env = en.asr(ATTACK_SECONDS, 1, RELEASE_SECONDS, gate);
dry = organ * env * volume;
reverb(x) = x : re.mono_freeverb(reverbSize, reverbDamp, REVERB_SPREAD, REVERB_CUTOFF_HZ);
output = dry * (1 - reverbAmount * REVERB_DRY_SCALE) + reverb(dry) * reverbAmount;

process = output;

effect = _,_;
"""

_CHURCH_ORGAN_SOURCE = r"""declare name "churchOrgan";
declare author "Remi Chapelle";

import("stdfaust.lib");

f = hslider("[00]freq[unit:Hz]",440,50,1000,0.1);
g = hslider("[01]gain",1,0,1,0.01);
t = button("[10]gate") : si.smoo;
p8 = hslider("[03]gain 8ve partial",1,0,1,0.01);
p5 = hslider("[04]gain 5th partial",1,0,1,0.01);
p3 = hslider("[05]gain 3d partial",1,0,1,0.01);
px = hslider("[06]gain other partials",0.05,0,1,0.01);
p0 = hslider("[02]gain fundamental",1,0,1,0.01);
psub = hslider("[07]gain lower octave",1,0,1,0.01);
nog = hslider("[08]noise gain",0.01,0,1,0.001);
pg = hslider("[09]gain preset",1,0,1,0.01);

r = dm.zita_light;

orgue = os.osc(f)       *p0
        + os.osc(f*2)   *p8
        + os.osc(f*0.5) *psub*0.5
        + os.osc(f*1.5) *p5*0.3
        + os.osc(f*3)   *p5*0.9
        + os.osc(f*4)   *p8*0.8
        + os.osc(f*5)   *p3*0.7
        + os.osc(f*6)   *p5*0.6
        + os.osc(f*7)   *px*0.5
        + os.osc(f*8)   *p8*0.6
        + os.osc(f*9)   *px*0.3
        + os.osc(f*10)  *p3*0.2
        + os.osc(f*11)  *px*0.15
        + os.osc(f*12)  *p5*0.1
        + os.osc(f*13)  *px*0.8
        + os.osc(f*14)  *px*0.6
        + os.osc(f*15)  *px*0.5
        + os.osc(f*16)  *p8*0.4
        + no.noise*nog;

process = orgue*g*t <: r;
"""

_FILTER_OSC_SOURCE = r"""declare name "filterOSC";
declare version "0.0";
declare author "JOS, revised by RM";
declare description "Simple application demoing filter based oscillators.";

import("stdfaust.lib");

process = dm.oscrs_demo;
"""

_BUBBLE_SOURCE = r"""declare name "bubble";
declare description "Production of a water drop bubble sound.";
declare license "MIT";
declare copyright "(c) 2017: Yann Orlarey, GRAME";

import("stdfaust.lib");

bubble(f0,trig) = os.osc(f) * (exp(-damp*time) : si.smooth(0.99))
    with {
        damp = 0.043*f0 + 0.0014*f0^(3/2);
        f = f0*(1+sigma*time);
        sigma = eta * damp;
        eta = 0.075;
        time = 0 : (select2(trig>trig'):+(1)) ~ _ : ba.samp2sec;
    };

process = button("drop") : bubble(hslider("v:bubble/freq", 600, 150, 2000, 1)) <: dm.freeverb_demo;
"""

_AUGMENTOR_SOURCE = r'''import("stdfaust.lib");

declare name "augmentor";
declare author "synth-setter";
declare description "Augmentation voice: random envelope, pitch shift, resonant filter, noise, and block reverse around a gated oscillator.";
declare license "MIT";

TRANSPOSE_WINDOW = 1024;
TRANSPOSE_XFADE = 64;
MIN_Q = 0.5;
Q_RANGE = 7.5;
NOISE_GAIN = 0.25;
REVERSE_WINDOW = 22050;
REVERSE_MAX_DELAY = 48000;
OUTPUT_GAIN = 0.8;

freq = hslider("v:augmentor/source/freq[unit:Hz]", 220, 50, 1000, 0.1);
srcGain = hslider("v:augmentor/source/gain", 0.5, 0, 1, 0.01);
gate = button("v:augmentor/gate");
voice = os.osc(freq) * srcGain * en.asr(0.01, 1, 0.1, gate);

// Wandering amplitude from rectified, smoothed noise.
envRate = hslider("v:augmentor/envelope/rate[unit:Hz]", 4, 0.1, 30, 0.1);
envDepth = hslider("v:augmentor/envelope/depth", 0.4, 0, 1, 0.01);
randMod = abs(no.noise) : fi.lowpass(1, envRate);
enveloped = voice * ((1 - envDepth) + envDepth * randMod);

// Stock faustlibraries pitch shifter (misceffects.lib).
shift = hslider("v:augmentor/pitch/shift[unit:semitones]", 0, -12, 12, 0.1);
pitchMix = hslider("v:augmentor/pitch/mix", 0, 0, 1, 0.01);
shifted = enveloped : ef.transpose(TRANSPOSE_WINDOW, TRANSPOSE_XFADE, shift);
pitched = enveloped * (1 - pitchMix) + shifted * pitchMix;

// Resonant lowpass; Q stays above zero so the 1/Q pole never blows up.
cutoff = hslider("v:augmentor/filter/cutoff[unit:Hz]", 4000, 100, 8000, 1);
reso = hslider("v:augmentor/filter/resonance", 0.3, 0, 1, 0.01);
filtMix = hslider("v:augmentor/filter/mix", 0, 0, 1, 0.01);
filtered = pitched : fi.resonlp(cutoff, MIN_Q + reso * Q_RANGE, 1);
filtOut = pitched * (1 - filtMix) + filtered * filtMix;

noiseAmt = hslider("v:augmentor/noise/amount", 0, 0, 1, 0.01);
noised = filtOut + no.noise * noiseAmt * NOISE_GAIN;

// Block reverser with one-window latency: output(t) = input(t - 2p - 1).
revMix = hslider("v:augmentor/reverse/mix", 0, 0, 1, 0.01);
pos = (+(1)) ~ _ : %(REVERSE_WINDOW);
reversed = noised : de.delay(REVERSE_MAX_DELAY, int(2 * pos + 1));
outMix = noised * (1 - revMix) + reversed * revMix;

process = outMix * OUTPUT_GAIN <: _,_;
'''

_faust_dsps: dict[ParamSpecName, FaustDsp] = {
    ParamSpecName("faust_augmentor"): FaustDsp(_AUGMENTOR_SOURCE, num_voices=0, outputs=2),
    ParamSpecName("faust_bright_organ"): FaustDsp(_BRIGHT_ORGAN_SOURCE, num_voices=1, outputs=2),
    ParamSpecName("faust_bubble"): FaustDsp(_BUBBLE_SOURCE, num_voices=0, outputs=2),
    ParamSpecName("faust_church_organ"): FaustDsp(_CHURCH_ORGAN_SOURCE, num_voices=0, outputs=2),
    ParamSpecName("faust_filter_osc"): FaustDsp(_FILTER_OSC_SOURCE, num_voices=0, outputs=1),
    ParamSpecName("faust_shimmer_fdn"): FaustDsp(SHIMMER_FDN_SOURCE, num_voices=0, outputs=2),
}
faust_dsps = cast(Mapping[str, FaustDsp], MappingProxyType(_faust_dsps))


def resolve_faust_dsp(param_spec_name: ParamSpecName) -> FaustDsp:
    """Resolve one checked-in source string without filesystem or URI fallback.

    :param param_spec_name: Shared Faust source and parameter-spec registry key.
    :returns: Registered in-memory Faust program.
    :raises KeyError: If the key has no checked-in Faust source.
    """
    try:
        return _faust_dsps[param_spec_name]
    except KeyError:
        raise KeyError(param_spec_name) from None
