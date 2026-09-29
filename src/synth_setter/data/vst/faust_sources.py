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

    .. attribute :: inputs

       Native input channel count.
    """

    source: str
    num_voices: int
    outputs: int
    inputs: int = 0


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

_FDN_EFFECT_SOURCE = r'''declare name "fdnEffect";

import("stdfaust.lib");

decay = hslider("decay", 0.7, 0.0, 0.95, 0.01);
damping = hslider("damping", 6000, 500, 18000, 1);
dryWet = hslider("dryWet", 0.5, 0.0, 1.0, 0.01);

comb(delay) = + ~ (@(delay) : fi.lowpass(1, damping) : *(decay));
fdn = _ <: comb(149), comb(211), comb(263), comb(293) :> /(4);
effect = _ <: *(1-dryWet), (fdn : *(dryWet)) :> _;

process = effect, effect;
'''

_FILTER_OSC_SOURCE = r"""declare name "filterOSC";
declare version "0.0";
declare author "JOS, revised by RM";
declare description "Simple application demoing filter based oscillators.";

import("stdfaust.lib");

process = dm.oscrs_demo;
"""

_KRONECKER_FDN_SOURCE = r'''import("stdfaust.lib");
declare name "kroneckerFDN";

t60_dc = hslider("Decay/t60_dc [unit:s]", 1.5, 0.1, 4.0, 0.01);
t60_nyquist = hslider("Decay/t60_nyquist [unit:s]", 1.5, 0.1, 4.0, 0.01);

// Shelf crossover mirrors the 6 kHz pyFDN build target.
sh_fc = min(6000, ma.SR / 5);
sh_t = tan(sh_fc / ma.SR * 2 * ma.PI);

d0 = hslider("Delays/d0 [unit:samp]", 601, 400, 1200, 1);
d1 = hslider("Delays/d1 [unit:samp]", 773, 400, 1200, 1);
d2 = hslider("Delays/d2 [unit:samp]", 839, 400, 1200, 1);
d3 = hslider("Delays/d3 [unit:samp]", 911, 400, 1200, 1);
d4 = hslider("Delays/d4 [unit:samp]", 997, 400, 1200, 1);
d5 = hslider("Delays/d5 [unit:samp]", 1063, 400, 1200, 1);
d6 = hslider("Delays/d6 [unit:samp]", 1129, 400, 1200, 1);
d7 = hslider("Delays/d7 [unit:samp]", 1181, 400, 1200, 1);

b0 = hslider("Input/b0", 0.5, -1, 1, 0.001);
b1 = hslider("Input/b1", 0.5, -1, 1, 0.001);
b2 = hslider("Input/b2", 0.5, -1, 1, 0.001);
b3 = hslider("Input/b3", 0.5, -1, 1, 0.001);
b4 = hslider("Input/b4", 0.5, -1, 1, 0.001);
b5 = hslider("Input/b5", 0.5, -1, 1, 0.001);
b6 = hslider("Input/b6", 0.5, -1, 1, 0.001);
b7 = hslider("Input/b7", 0.5, -1, 1, 0.001);

c0 = hslider("Output/c0", 0.125, -1, 1, 0.001);
c1 = hslider("Output/c1", 0.125, -1, 1, 0.001);
c2 = hslider("Output/c2", 0.125, -1, 1, 0.001);
c3 = hslider("Output/c3", 0.125, -1, 1, 0.001);
c4 = hslider("Output/c4", 0.125, -1, 1, 0.001);
c5 = hslider("Output/c5", 0.125, -1, 1, 0.001);
c6 = hslider("Output/c6", 0.125, -1, 1, 0.001);
c7 = hslider("Output/c7", 0.125, -1, 1, 0.001);
dry = hslider("Output/dry", 0, -1, 1, 0.001);

a0 = hslider("Kernel/a0 [unit:rad]", 0.7853982, -3.14159, 3.14159, 0.001);
a1 = hslider("Kernel/a1 [unit:rad]", 0.7853982, -3.14159, 3.14159, 0.001);
a2 = hslider("Kernel/a2 [unit:rad]", 0.7853982, -3.14159, 3.14159, 0.001);
r0 = checkbox("Kernel/r0");
r1 = checkbox("Kernel/r1");
r2 = checkbox("Kernel/r2");

cA0 = cos(a0); sA0 = sin(a0);
cA1 = cos(a1); sA1 = sin(a1);
cA2 = cos(a2); sA2 = sin(a2);
k0x(x1, x2) = (1 - r0) * (cA0 * x1 - sA0 * x2) + r0 * (cA0 * x1 + sA0 * x2);
k0y(x1, x2) = (1 - r0) * (sA0 * x1 + cA0 * x2) + r0 * (sA0 * x1 - cA0 * x2);
k1x(x1, x2) = (1 - r1) * (cA1 * x1 - sA1 * x2) + r1 * (cA1 * x1 + sA1 * x2);
k1y(x1, x2) = (1 - r1) * (sA1 * x1 + cA1 * x2) + r1 * (sA1 * x1 - cA1 * x2);
k2x(x1, x2) = (1 - r2) * (cA2 * x1 - sA2 * x2) + r2 * (cA2 * x1 + sA2 * x2);
k2y(x1, x2) = (1 - r2) * (sA2 * x1 + cA2 * x2) + r2 * (sA2 * x1 - cA2 * x2);

// Per-line first-order shelf: same endpoint math as the pyFDN build.
gdc0 = pow(0.001, d0 / (t60_dc * ma.SR));
gny0 = pow(0.001, d0 / (t60_nyquist * ma.SR));
shsq0 = sqrt(gdc0 / gny0);
shn0 = sh_t / shsq0 + 1;
shb0_0 = (sh_t * shsq0 + 1) * gny0 / shn0;
shb1_0 = (sh_t * shsq0 - 1) * gny0 / shn0;
sha1_0 = (sh_t / shsq0 - 1) / shn0;
gdc1 = pow(0.001, d1 / (t60_dc * ma.SR));
gny1 = pow(0.001, d1 / (t60_nyquist * ma.SR));
shsq1 = sqrt(gdc1 / gny1);
shn1 = sh_t / shsq1 + 1;
shb0_1 = (sh_t * shsq1 + 1) * gny1 / shn1;
shb1_1 = (sh_t * shsq1 - 1) * gny1 / shn1;
sha1_1 = (sh_t / shsq1 - 1) / shn1;
gdc2 = pow(0.001, d2 / (t60_dc * ma.SR));
gny2 = pow(0.001, d2 / (t60_nyquist * ma.SR));
shsq2 = sqrt(gdc2 / gny2);
shn2 = sh_t / shsq2 + 1;
shb0_2 = (sh_t * shsq2 + 1) * gny2 / shn2;
shb1_2 = (sh_t * shsq2 - 1) * gny2 / shn2;
sha1_2 = (sh_t / shsq2 - 1) / shn2;
gdc3 = pow(0.001, d3 / (t60_dc * ma.SR));
gny3 = pow(0.001, d3 / (t60_nyquist * ma.SR));
shsq3 = sqrt(gdc3 / gny3);
shn3 = sh_t / shsq3 + 1;
shb0_3 = (sh_t * shsq3 + 1) * gny3 / shn3;
shb1_3 = (sh_t * shsq3 - 1) * gny3 / shn3;
sha1_3 = (sh_t / shsq3 - 1) / shn3;
gdc4 = pow(0.001, d4 / (t60_dc * ma.SR));
gny4 = pow(0.001, d4 / (t60_nyquist * ma.SR));
shsq4 = sqrt(gdc4 / gny4);
shn4 = sh_t / shsq4 + 1;
shb0_4 = (sh_t * shsq4 + 1) * gny4 / shn4;
shb1_4 = (sh_t * shsq4 - 1) * gny4 / shn4;
sha1_4 = (sh_t / shsq4 - 1) / shn4;
gdc5 = pow(0.001, d5 / (t60_dc * ma.SR));
gny5 = pow(0.001, d5 / (t60_nyquist * ma.SR));
shsq5 = sqrt(gdc5 / gny5);
shn5 = sh_t / shsq5 + 1;
shb0_5 = (sh_t * shsq5 + 1) * gny5 / shn5;
shb1_5 = (sh_t * shsq5 - 1) * gny5 / shn5;
sha1_5 = (sh_t / shsq5 - 1) / shn5;
gdc6 = pow(0.001, d6 / (t60_dc * ma.SR));
gny6 = pow(0.001, d6 / (t60_nyquist * ma.SR));
shsq6 = sqrt(gdc6 / gny6);
shn6 = sh_t / shsq6 + 1;
shb0_6 = (sh_t * shsq6 + 1) * gny6 / shn6;
shb1_6 = (sh_t * shsq6 - 1) * gny6 / shn6;
sha1_6 = (sh_t / shsq6 - 1) / shn6;
gdc7 = pow(0.001, d7 / (t60_dc * ma.SR));
gny7 = pow(0.001, d7 / (t60_nyquist * ma.SR));
shsq7 = sqrt(gdc7 / gny7);
shn7 = sh_t / shsq7 + 1;
shb0_7 = (sh_t * shsq7 + 1) * gny7 / shn7;
shb1_7 = (sh_t * shsq7 - 1) * gny7 / shn7;
sha1_7 = (sh_t / shsq7 - 1) / shn7;

// Feedback ports lead: Faust connects `~` outputs to the first inputs.
// Input and delay lines are offset by one sample each way so the
// implicit `~` sample lands exactly on the pyFDN recirculation grid.
F(f0, f1, f2, f3, f4, f5, f6, f7, x) = z0, z1, z2, z3, z4, z5, z6, z7 with {
  u0 = b0 * x' + f0;
  s0 = de.delay(2048, int(d0) - 1, u0);
  z0 = s0 : fi.tf1(shb0_0, shb1_0, sha1_0);
  u1 = b1 * x' + f1;
  s1 = de.delay(2048, int(d1) - 1, u1);
  z1 = s1 : fi.tf1(shb0_1, shb1_1, sha1_1);
  u2 = b2 * x' + f2;
  s2 = de.delay(2048, int(d2) - 1, u2);
  z2 = s2 : fi.tf1(shb0_2, shb1_2, sha1_2);
  u3 = b3 * x' + f3;
  s3 = de.delay(2048, int(d3) - 1, u3);
  z3 = s3 : fi.tf1(shb0_3, shb1_3, sha1_3);
  u4 = b4 * x' + f4;
  s4 = de.delay(2048, int(d4) - 1, u4);
  z4 = s4 : fi.tf1(shb0_4, shb1_4, sha1_4);
  u5 = b5 * x' + f5;
  s5 = de.delay(2048, int(d5) - 1, u5);
  z5 = s5 : fi.tf1(shb0_5, shb1_5, sha1_5);
  u6 = b6 * x' + f6;
  s6 = de.delay(2048, int(d6) - 1, u6);
  z6 = s6 : fi.tf1(shb0_6, shb1_6, sha1_6);
  u7 = b7 * x' + f7;
  s7 = de.delay(2048, int(d7) - 1, u7);
  z7 = s7 : fi.tf1(shb0_7, shb1_7, sha1_7);
};

B(z0, z1, z2, z3, z4, z5, z6, z7) = w0, w1, w2, w3, w4, w5, w6, w7 with {
  m0 = k0x(z0, z1); m1 = k0y(z0, z1);
  m2 = k0x(z2, z3); m3 = k0y(z2, z3);
  m4 = k0x(z4, z5); m5 = k0y(z4, z5);
  m6 = k0x(z6, z7); m7 = k0y(z6, z7);
  n0 = k1x(m0, m2); n2 = k1y(m0, m2);
  n1 = k1x(m1, m3); n3 = k1y(m1, m3);
  n4 = k1x(m4, m6); n6 = k1y(m4, m6);
  n5 = k1x(m5, m7); n7 = k1y(m5, m7);
  w0 = k2x(n0, n4); w4 = k2y(n0, n4);
  w1 = k2x(n1, n5); w5 = k2y(n1, n5);
  w2 = k2x(n2, n6); w6 = k2y(n2, n6);
  w3 = k2x(n3, n7); w7 = k2y(n3, n7);
};

// `~` contributes one implicit sample of delay on the feedback path.
loop = F ~ B;
pg = _ * c0, _ * c1, _ * c2, _ * c3, _ * c4, _ * c5, _ * c6, _ * c7;
fdn = _ <: ((_ * dry), (loop : (pg :> _))) :> _;
process = 1 - 1' : fdn;
'''

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

_faust_dsps: dict[ParamSpecName, FaustDsp] = {
    ParamSpecName("faust_bright_organ"): FaustDsp(_BRIGHT_ORGAN_SOURCE, num_voices=1, outputs=2),
    ParamSpecName("faust_bubble"): FaustDsp(_BUBBLE_SOURCE, num_voices=0, outputs=2),
    ParamSpecName("faust_church_organ"): FaustDsp(_CHURCH_ORGAN_SOURCE, num_voices=0, outputs=2),
    ParamSpecName("faust_fdn_effect"): FaustDsp(
        _FDN_EFFECT_SOURCE, num_voices=0, inputs=2, outputs=2
    ),
    ParamSpecName("faust_filter_osc"): FaustDsp(_FILTER_OSC_SOURCE, num_voices=0, outputs=1),
    ParamSpecName("faust_kronecker_fdn"): FaustDsp(
        _KRONECKER_FDN_SOURCE, num_voices=0, outputs=1
    ),
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
