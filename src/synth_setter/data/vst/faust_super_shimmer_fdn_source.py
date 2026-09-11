"""Checked-in Faust 16-line super-shimmer FDN source."""

SUPER_SHIMMER_FDN_SOURCE = r'''declare name        "superShimmerFDN";
declare version     "1.0";
declare author      "synth-setter contributors";
declare license     "CC BY 4.0 AND MIT";
declare description "Sixteen-line modulated shimmer and granular feedback delay network.";

import("stdfaust.lib");

N = 16;

delays = (
    1103, 1181, 1399, 1597, 1709, 2933, 3690, 4301,
    4871, 5237, 5801, 6353, 6947, 7481, 8011, 8623
);
delay(i) = ba.take(i+1, delays);
maxLine = 16384;

// H2 tensor power four, normalized by sqrt(16), is orthogonal.
feedbackMatrix = (
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25,
    0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25,
    0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25,
    0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25,
    0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25,
    0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25,
    0.25, -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25,
    0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25,
    0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25,
    0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25,
    0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25,
    0.25, 0.25, 0.25, 0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25,
    0.25, -0.25, 0.25, -0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, 0.25, -0.25,
    0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, -0.25, -0.25,
    0.25, -0.25, -0.25, 0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, 0.25, -0.25, -0.25, 0.25
);
A(i, j) = ba.take(i*N + j + 1, feedbackMatrix);
mixingMatrix = si.bus(N) <: par(i, N, (par(j, N, *(A(i, j))) :> _));

inGain = 1.0 / sqrt(N);
outGain = 1.0 / sqrt(N);

t60Low = hslider("h:[0]FDN/[0]T60 low  [unit:s][style:knob]", 4.4, 0.1, 20, 0.1);
t60High = hslider("h:[0]FDN/[1]T60 high [unit:s][style:knob]", 0.5, 0.05, 20, 0.05);
fc = hslider("h:[0]FDN/[2]crossover [unit:Hz][style:knob][scale:log]", 4000, 200, 16000, 1);

t60gain(m, t60) = pow(10.0, -3.0 * m / (ma.SR * t60));
attenuation(i) = *(gLow) : fi.highshelf(1, 20.0 * log10(gHigh / gLow), fc)
with {
    gLow = t60gain(delay(i), t60Low);
    gHigh = t60gain(delay(i), t60High);
};

lfoRate = hslider("h:[1]Modulation/[0]rate [unit:Hz][style:knob][scale:log]", 0.1, 0.01, 0.5, 0.01);
lfoDepth = hslider("h:[1]Modulation/[1]depth [unit:samples][style:knob]", 64, 0, 256, 1);

cubicRead(maxDelay, d, x) = ((a * f + b) * f + c) * f + y1
with {
    i = int(d);
    f = d - i;
    y0 = x : de.delay(maxDelay, i - 1);
    y1 = x : de.delay(maxDelay, i);
    y2 = x : de.delay(maxDelay, i + 1);
    y3 = x : de.delay(maxDelay, i + 2);
    a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    c = -0.5 * y0 + 0.5 * y2;
};

modulatedDelay(i, x) = cubicRead(maxLine, delay(i) + lfoDepth * os.oscp(lfoRate, 2.0*ma.PI*i/N), x);

compMaxGain = hslider("h:[2]Shimmer/[3]DC comp max [unit:dB][style:knob]", 0, 0, 12, 0.1) : ba.db2linear;

dcBlockerComp(x) = y * g
with {
    R = 0.995;
    eps = 1e-12;
    alphaEnv = exp(-1.0 / (ma.SR * 0.05));
    alphaGain = exp(-1.0 / (ma.SR * 0.02));
    y = x : fi.zero(1.0) : fi.pole(R);
    pIn = x * x : si.smooth(alphaEnv);
    pOut = y * y : si.smooth(alphaEnv);
    g = min(sqrt((pIn + eps) / (pOut + eps)), compMaxGain) - 1.0 : si.smooth(alphaGain) : +(1.0);
};

maxWindow = 8192;
maxPitchDelay = maxWindow + 8;
minPitchDelay = 3;
transposeCents = hslider("h:[2]Shimmer/[0]transpose [unit:cents][style:knob]", -700, -2400, 2400, 1);
windowSize = hslider("h:[2]Shimmer/[1]window [unit:samples][style:knob][scale:log]", 2048, 64, maxWindow, 1);
pitchRatio = pow(2.0, transposeCents / 1200.0);
phaseInc = (1.0 - pitchRatio) / windowSize;
phase1 = ((+(phaseInc) : ma.frac) ~ _) : mem;
phase2 = phase1 + 0.5 : ma.frac;

energyGuardOn = 1 - checkbox("h:[5]Safety/[1]energy guard bypass");
energyGuard(x, y) = y * g
with {
    eps = 1e-12;
    alphaEnv = exp(-1.0 / (ma.SR * 0.05));
    alphaGain = exp(-1.0 / (ma.SR * 0.02));
    pIn = x * x : si.smooth(alphaEnv);
    pOut = y * y : si.smooth(alphaEnv);
    gRaw = min(1.0, sqrt((pIn + eps) / (pOut + eps)));
    g = select2(energyGuardOn, 1.0, gRaw) - 1.0 : si.smooth(alphaGain) : +(1.0);
};

pitchShift(x) = energyGuard(x, shifted)
with {
    head(ph) = cubicRead(maxPitchDelay, minPitchDelay + ph * windowSize, x) * sin(ma.PI * ph);
    shifted = (head(phase1) + head(phase2)) : dcBlockerComp;
};
active(i) = nentry("h:[2]Shimmer/h:[2]shifted lines/line %2i [style:knob]", (i >= 6), 0, 1, 1);
shiftLine(i, x) = select2(active(i), x, pitchShift(x));

grainDuration = hslider("h:[3]Granular/[0]duration [unit:s][style:knob]", 0.08, 0.04, 0.12, 0.001);
grainPosition = hslider("h:[3]Granular/[1]position [unit:s][style:knob]", 0.15, 0.08, 0.25, 0.001);
grainJitter = hslider("h:[3]Granular/[2]jitter [unit:s][style:knob]", 0.005, 0, 0.02, 0.001);

// Exact MIT granular implementation and Hann helper closure from faustlibraries
// 271228a08981fa10b07732f0861421d1e20d4022; DawDreamer 0.8.3 predates both APIs.
declare granular license "MIT";
windowCosN(coeffs, x) = sum(k, K, ba.take(k+1, coeffs) * cos(k*ma.PI*2.0*x))
                         * ((x >= 0.0) & (x <= 1.0))
with { K = ba.count(coeffs); };
windowHann(x) = windowCosN((0.5, -0.5), x);
granular(P, dur, ratio, pos, jit, sig) = sum(i, P, voice(i)) * (2.0/float(P))
with {
    maxDelay = 65536;
    durSamp = max(1.0, dur*ma.SR);
    ph = (+(1.0/durSamp) ~ ma.frac);
    phi(i) = ma.frac(ph + float(i)/float(P));
    trig(i) = phi(i) < phi(i)';
    start(i) = ba.sAndH(trig(i), (pos + jit*abs(no.noises(P, i)))*ma.SR);
    dly(i) = start(i) + (1.0 - ratio)*phi(i)*durSamp;
    voice(i) = de.fdelay(maxDelay, max(0.0, min(float(maxDelay-1), dly(i))), sig)
               * windowHann(phi(i));
};

granularLine(x) = energyGuard(x, granular(4, grainDuration, 1, grainPosition, grainJitter, x));
lineEffect(i, x) = select2(i >= 8, shiftLine(i, x), granularLine(x));

loopCeiling = hslider("h:[5]Safety/[0]loop ceiling [unit:dB][style:knob]", -12, -40, 0, 0.5) : ba.db2linear;
loopLimiter = si.bus(N) <: (si.bus(N), gain) : ro.interleave(N, 2) : par(i, N, *)
with {
    env = par(i, N, *(_)) :> /(N) : an.amp_follower_ud(0.0005, 0.05) : sqrt;
    gain = min(1.0, loopCeiling / max(env, 1e-9)) <: si.bus(N);
};

safeOut(x) = max(-1.0, min(1.0, select2((x != x) | (abs(x) > 1e6), x, 0.0)));
lineFilter(i) = modulatedDelay(i) : attenuation(i) : lineEffect(i);
fdn = (_ <: par(i, N, *(inGain)))
    : ((ro.interleave(N, 2) : par(i, N, +) : par(i, N, lineFilter(i))) ~ (loopLimiter : mixingMatrix))
    : par(i, N, *(outGain)) :> _;

dryWet = hslider("h:[4]Output/[0]dry/wet [style:knob]", 0.5, 0, 1, 0.01);
outLvl = hslider("h:[4]Output/[1]level [unit:dB][style:knob]", 0, -40, 12, 0.1) : ba.db2linear;
superShimmer = _ <: *(1 - dryWet), (fdn * dryWet) :> *(outLvl) : safeOut <: _, _;
process = ba.impulsify(1.0) : superShimmer;
'''
