"""Checked-in CC BY 4.0 Faust shimmer FDN source."""

SHIMMER_FDN_SOURCE = r"""declare name        "shimmerFDN";
declare version     "1.0";
declare author      "Faust port of pyFDN example_shimmer_fdn (td.PitchShift)";
declare license     "CC BY 4.0";
declare description "Pitch-shifting (shimmer) feedback delay network. Faust port of the
    'Pitch Shift' operator of the pyFDN marimo notebook example_shimmer_fdn and of the
    ring-buffer time-compression nonlinearity (Sec. 3.4 / 4.1) in Dal Santo, Pi, Prawda,
    Schlecht, Valimaki, 'Shimmer Reverberation with Nonlinear Feedback Delay Networks',
    Proc. DAFx26, Cambridge MA, 2026.";

import("stdfaust.lib");

// Constants follow the cited pyFDN shimmer example and DAFx26 formulation.

N = 8;

delays   = (1103, 1181, 1399, 1597, 1709, 2933, 3690, 4301);
delay(i) = ba.take(i+1, delays);
maxLine  = 8192;

feedbackMatrix = (
    -0.142739, -0.028149, -0.746489, -0.077929, -0.434721,  0.112789,  0.133279, -0.442773,
    -0.362347, -0.504027, -0.065965,  0.372110, -0.180097, -0.617614,  0.061826,  0.232681,
    -0.006552,  0.663555,  0.037719,  0.182554, -0.580807, -0.165896, -0.240269,  0.319868,
    -0.116046, -0.130404, -0.282667, -0.676789, -0.014137,  0.043422, -0.040571,  0.654104,
    -0.833904,  0.084785,  0.090393,  0.055160,  0.150670,  0.332090, -0.383683, -0.077493,
    -0.107376,  0.438025, -0.448410,  0.289616,  0.577090, -0.124813,  0.341110,  0.216076,
    -0.283977,  0.056719,  0.338394, -0.032535, -0.272885,  0.262388,  0.804513,  0.100083,
    -0.217377,  0.292544,  0.182558, -0.525464,  0.104145, -0.617831,  0.090717, -0.396148
);

A(i, j) = ba.take(i*N + j + 1, feedbackMatrix);

mixingMatrix = si.bus(N) <: par(i, N, (par(j, N, *(A(i, j))) :> _));

inGain  = 1.0 / sqrt(N);
outGain = 1.0 / sqrt(N);

t60Low  = hslider("h:[0]FDN/[0]T60 low  [unit:s][style:knob]",  4.4, 0.1, 20, 0.1);
t60High = hslider("h:[0]FDN/[1]T60 high [unit:s][style:knob]",  0.5, 0.05, 20, 0.05);
fc      = hslider("h:[0]FDN/[2]crossover [unit:Hz][style:knob][scale:log]", 4000, 200, 16000, 1);

t60gain(m, t60) = pow(10.0, -3.0 * m / (ma.SR * t60));

attenuation(i) = *(gLow) : fi.highshelf(1, 20.0 * log10(gHigh / gLow), fc)
with {
    gLow  = t60gain(delay(i), t60Low);
    gHigh = t60gain(delay(i), t60High);
};

compMaxGain = hslider("h:[1]Shimmer/[3]DC comp max [unit:dB][style:knob]", 0, 0, 12, 0.1) : ba.db2linear;

dcBlockerComp(x) = y * g
with {
    R         = 0.995;
    eps       = 1e-12;
    maxGain   = compMaxGain;
    alphaEnv  = exp(-1.0 / (ma.SR * 0.05));
    alphaGain = exp(-1.0 / (ma.SR * 0.02));
    y    = x : fi.zero(1.0) : fi.pole(R);
    pIn  = x * x : si.smooth(alphaEnv);
    pOut = y * y : si.smooth(alphaEnv);

    g    = min(sqrt((pIn + eps) / (pOut + eps)), maxGain) - 1.0 : si.smooth(alphaGain) : +(1.0);
};

maxWindow = 8192;
maxDelay  = maxWindow + 8;
minDelay  = 3;

transposeCents = hslider("h:[1]Shimmer/[0]transpose [unit:cents][style:knob]", -700, -2400, 2400, 1);
windowSize     = hslider("h:[1]Shimmer/[1]window [unit:samples][style:knob][scale:log]", 2048, 64, maxWindow, 1);

pitchRatio = pow(2.0, transposeCents / 1200.0);
phaseInc   = (1.0 - pitchRatio) / windowSize;

phase1 = ((+(phaseInc) : ma.frac) ~ _) : mem;
phase2 = phase1 + 0.5 : ma.frac;

cubicRead(d, x) = ((a * f + b) * f + c) * f + y1
with {
    i  = int(d);
    f  = d - i;
    y0 = x : de.delay(maxDelay, i - 1);
    y1 = x : de.delay(maxDelay, i);
    y2 = x : de.delay(maxDelay, i + 1);
    y3 = x : de.delay(maxDelay, i + 2);
    a  = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    b  =        y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    c  = -0.5 * y0             + 0.5 * y2;
};

// Coherent pitch-head signals require a non-amplifying per-line energy guard.
energyGuardOn = 1 - checkbox("h:[3]Safety/[1]energy guard bypass");

energyGuard(x, y) = y * g
with {
    eps       = 1e-12;
    alphaEnv  = exp(-1.0 / (ma.SR * 0.05));
    alphaGain = exp(-1.0 / (ma.SR * 0.02));
    pIn  = x * x : si.smooth(alphaEnv);
    pOut = y * y : si.smooth(alphaEnv);
    gRaw = min(1.0, sqrt((pIn + eps) / (pOut + eps)));
    g    = select2(energyGuardOn, 1.0, gRaw) - 1.0 : si.smooth(alphaGain) : +(1.0);
};

pitchShift(x) = energyGuard(x, shifted)
with {
    head(ph) = cubicRead(minDelay + ph * windowSize, x) * sin(ma.PI * ph);
    shifted  = (head(phase1) + head(phase2)) : dcBlockerComp;
};

active(i)    = nentry("h:[1]Shimmer/h:[2]shifted lines/line %2i [style:knob]", (i >= N - 2), 0, 1, 1);
shiftLine(i) = _ <: select2(active(i), _, pitchShift);

// Loop and output guards bound nonlinear feedback and reject non-finite samples.
loopCeiling = hslider("h:[3]Safety/[0]loop ceiling [unit:dB][style:knob]", -12, -40, 0, 0.5) : ba.db2linear;

loopLimiter = si.bus(N) <: (si.bus(N), gain) : ro.interleave(N, 2) : par(i, N, *)
with {

    env  = par(i, N, *(_)) :> /(N) : an.amp_follower_ud(0.0005, 0.05) : sqrt;
    gain = min(1.0, loopCeiling / max(env, 1e-9)) <: si.bus(N);
};

safeOut(x) = max(-1.0, min(1.0, select2((x != x) | (abs(x) > 1e6), x, 0.0)));

lineFilter(i) = de.delay(maxLine, delay(i) - 1) : attenuation(i) : shiftLine(i);

fdn = (_ <: par(i, N, *(inGain)))
    : ((ro.interleave(N, 2) : par(i, N, +) : par(i, N, lineFilter(i))) ~ (loopLimiter : mixingMatrix))
    : par(i, N, *(outGain)) :> _;

dryWet = hslider("h:[2]Output/[0]dry/wet [style:knob]", 0.5, 0, 1, 0.01);
outLvl = hslider("h:[2]Output/[1]level [unit:dB][style:knob]", 0, -40, 12, 0.1) : ba.db2linear;

shimmer = _ <: *(1 - dryWet), (fdn * dryWet) :> *(outLvl) : safeOut <: _, _;

// A constant rising edge produces one unit impulse at sample zero.
process = ba.impulsify(1.0) : shimmer;
"""
