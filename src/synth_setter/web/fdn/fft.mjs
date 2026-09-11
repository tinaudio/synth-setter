// Radix-2 FFT with a Bluestein wrapper so librosa's non-power-of-two frame lengths work.

function radix2InPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let size = 2; size <= n; size <<= 1) {
    const angle = (-2 * Math.PI) / size;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    for (let start = 0; start < n; start += size) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < size / 2; k++) {
        const a = start + k;
        const b = a + size / 2;
        const productRe = re[b] * curRe - im[b] * curIm;
        const productIm = re[b] * curIm + im[b] * curRe;
        re[b] = re[a] - productRe;
        im[b] = im[a] - productIm;
        re[a] += productRe;
        im[a] += productIm;
        [curRe, curIm] = [curRe * wRe - curIm * wIm, curRe * wIm + curIm * wRe];
      }
    }
  }
}

function nextPowerOfTwo(value) {
  return 2 ** Math.ceil(Math.log2(value));
}

// Bluestein: an N-point DFT as one convolution of length ≥ 2N-1, done with radix-2 FFTs.
function bluestein(re, im) {
  const n = re.length;
  const m = nextPowerOfTwo(2 * n - 1);
  const chirpRe = new Float64Array(n);
  const chirpIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    // k*k mod 2n keeps the chirp phase exact for large frames.
    const angle = (Math.PI * ((k * k) % (2 * n))) / n;
    chirpRe[k] = Math.cos(angle);
    chirpIm[k] = -Math.sin(angle);
  }
  const aRe = new Float64Array(m);
  const aIm = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    aRe[k] = re[k] * chirpRe[k] - im[k] * chirpIm[k];
    aIm[k] = re[k] * chirpIm[k] + im[k] * chirpRe[k];
  }
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  bRe[0] = chirpRe[0];
  bIm[0] = -chirpIm[0];
  for (let k = 1; k < n; k++) {
    bRe[k] = bRe[m - k] = chirpRe[k];
    bIm[k] = bIm[m - k] = -chirpIm[k];
  }
  radix2InPlace(aRe, aIm);
  radix2InPlace(bRe, bIm);
  for (let k = 0; k < m; k++) {
    [aRe[k], aIm[k]] = [aRe[k] * bRe[k] - aIm[k] * bIm[k], aRe[k] * bIm[k] + aIm[k] * bRe[k]];
  }
  // Inverse FFT via conjugation.
  for (let k = 0; k < m; k++) aIm[k] = -aIm[k];
  radix2InPlace(aRe, aIm);
  for (let k = 0; k < n; k++) {
    const yRe = aRe[k] / m;
    const yIm = -aIm[k] / m;
    re[k] = yRe * chirpRe[k] - yIm * chirpIm[k];
    im[k] = yRe * chirpIm[k] + yIm * chirpRe[k];
  }
}

export function powerSpectrum(frame) {
  const n = frame.length;
  const re = Float64Array.from(frame);
  const im = new Float64Array(n);
  if ((n & (n - 1)) === 0) radix2InPlace(re, im);
  else bluestein(re, im);
  const bins = Math.floor(n / 2) + 1;
  const power = new Float64Array(bins);
  for (let bin = 0; bin < bins; bin++) power[bin] = re[bin] * re[bin] + im[bin] * im[bin];
  return power;
}
