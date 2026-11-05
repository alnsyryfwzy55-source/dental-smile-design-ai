/* Smile morphing on the patient's own photo. Pure pixel math on {w,h,data:Uint8ClampedArray RGBA}. Browser: window.DSDSim, Node: module.exports */
(function (root) {
  'use strict';
  const TEETH = ['13', '12', '11', '21', '22', '23'];
  const MIRROR = { '13': '23', '12': '22', '11': '21', '21': '11', '22': '12', '23': '13' };
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

  const makeImg = (w, h) => ({ w, h, data: new Uint8ClampedArray(w * h * 4) });
  function copyImg(a) { return { w: a.w, h: a.h, data: new Uint8ClampedArray(a.data) }; }

  function sample(img, x, y, out) {        // bilinear, clamped; out = [r,g,b]
    const w = img.w, h = img.h, d = img.data;
    x = clamp(x, 0, w - 1.001); y = clamp(y, 0, h - 1.001);
    const x0 = x | 0, y0 = y | 0, fx = x - x0, fy = y - y0, i = (y0 * w + x0) * 4, j = i + w * 4;
    for (let c = 0; c < 3; c++) out[c] = (d[i + c] * (1 - fx) + d[i + 4 + c] * fx) * (1 - fy) + (d[j + c] * (1 - fx) + d[j + 4 + c] * fx) * fy;
    return out;
  }

  // dst(u,v) = src(M·(u,v)), M = [a,b,c,d,e,f] (dst → src)
  function warpAffine(src, M, w, h) {
    const out = makeImg(w, h), px = [0, 0, 0];
    for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) {
      const x = M[0] * u + M[1] * v + M[2], y = M[3] * u + M[4] * v + M[5], o = (v * w + u) * 4;
      if (x < -1 || y < -1 || x > src.w || y > src.h) { out.data[o + 3] = 255; continue; }
      sample(src, x, y, px); out.data[o] = px[0]; out.data[o + 1] = px[1]; out.data[o + 2] = px[2]; out.data[o + 3] = 255;
    }
    return out;
  }

  // Paste a crop result back into the full image. Minv: image → crop. alpha: Float32 (crop size) blend weight.
  function pasteBack(dst, crop, alpha, Minv, M) {
    // bounding box of the crop in image coordinates
    const cs = [[0, 0], [crop.w, 0], [0, crop.h], [crop.w, crop.h]].map(([u, v]) => [M[0] * u + M[1] * v + M[2], M[3] * u + M[4] * v + M[5]]);
    const x0 = Math.max(0, Math.floor(Math.min(...cs.map((c) => c[0])))), x1 = Math.min(dst.w - 1, Math.ceil(Math.max(...cs.map((c) => c[0]))));
    const y0 = Math.max(0, Math.floor(Math.min(...cs.map((c) => c[1])))), y1 = Math.min(dst.h - 1, Math.ceil(Math.max(...cs.map((c) => c[1]))));
    const px = [0, 0, 0];
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const u = Minv[0] * x + Minv[1] * y + Minv[2], v = Minv[3] * x + Minv[4] * y + Minv[5];
      if (u < 0 || v < 0 || u > crop.w - 1 || v > crop.h - 1) continue;
      const a = sampleAlpha(alpha, crop.w, crop.h, u, v); if (a <= 0.002) continue;
      sample(crop, u, v, px);
      const o = (y * dst.w + x) * 4;
      for (let c = 0; c < 3; c++) dst.data[o + c] = dst.data[o + c] * (1 - a) + px[c] * a;
    }
    return dst;
  }
  function sampleAlpha(A, w, h, x, y) {
    const x0 = Math.min(w - 2, x | 0), y0 = Math.min(h - 2, y | 0), fx = x - x0, fy = y - y0, i = y0 * w + x0;
    return (A[i] * (1 - fx) + A[i + 1] * fx) * (1 - fy) + (A[i + w] * (1 - fx) + A[i + w + 1] * fx) * fy;
  }

  function blurAlpha(A, w, h, r) {           // separable box blur ×2 ≈ gaussian
    if (r <= 0) return Float32Array.from(A);
    let a = Float32Array.from(A), b = new Float32Array(w * h);
    for (let pass = 0; pass < 2; pass++) {
      for (let y = 0; y < h; y++) { let s = 0; for (let x = -r; x <= r; x++) s += a[y * w + clamp(x, 0, w - 1)];
        for (let x = 0; x < w; x++) { b[y * w + x] = s / (2 * r + 1); s += a[y * w + clamp(x + r + 1, 0, w - 1)] - a[y * w + clamp(x - r, 0, w - 1)]; } }
      for (let x = 0; x < w; x++) { let s = 0; for (let y = -r; y <= r; y++) s += b[clamp(y, 0, h - 1) * w + x];
        for (let y = 0; y < h; y++) { a[y * w + x] = s / (2 * r + 1); s += b[clamp(y + r + 1, 0, h - 1) * w + x] - b[clamp(y - r, 0, h - 1) * w + x]; } }
    }
    return a;
  }

  // Fill masked pixels (mask=1) smoothly from their surroundings (push–pull pyramid)
  function inpaint(img, mask) {
    const w0 = img.w, h0 = img.h, levels = [];
    let w = w0, h = h0, C = new Float32Array(w * h * 3), Wt = new Float32Array(w * h);
    for (let k = 0; k < w * h; k++) { if (mask[k]) continue; Wt[k] = 1; C[k * 3] = img.data[k * 4]; C[k * 3 + 1] = img.data[k * 4 + 1]; C[k * 3 + 2] = img.data[k * 4 + 2]; }
    levels.push({ w, h, C, Wt });
    while (w > 2 && h > 2) {
      const nw = Math.ceil(w / 2), nh = Math.ceil(h / 2), nC = new Float32Array(nw * nh * 3), nW = new Float32Array(nw * nh);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const k = y * w + x, q = (y >> 1) * nw + (x >> 1);
        nW[q] += Wt[k]; nC[q * 3] += C[k * 3]; nC[q * 3 + 1] += C[k * 3 + 1]; nC[q * 3 + 2] += C[k * 3 + 2];
      }
      for (let q = 0; q < nw * nh; q++) { const s = nW[q]; if (s > 0) { nC[q * 3] /= s; nC[q * 3 + 1] /= s; nC[q * 3 + 2] /= s; nW[q] = Math.min(1, s); } }
      w = nw; h = nh; C = nC; Wt = nW; levels.push({ w, h, C, Wt });
    }
    for (let l = levels.length - 2; l >= 0; l--) {
      const L = levels[l], U = levels[l + 1];
      for (let y = 0; y < L.h; y++) for (let x = 0; x < L.w; x++) {
        const k = y * L.w + x; if (L.Wt[k] >= 1) continue;
        const q = Math.min(U.h - 1, y >> 1) * U.w + Math.min(U.w - 1, x >> 1), a = L.Wt[k];
        for (let c = 0; c < 3; c++) L.C[k * 3 + c] = L.C[k * 3 + c] * a + U.C[q * 3 + c] * (1 - a);
        L.Wt[k] = 1;
      }
    }
    const out = copyImg(img), F = levels[0].C;
    for (let k = 0; k < w0 * h0; k++) if (mask[k]) { out.data[k * 4] = F[k * 3]; out.data[k * 4 + 1] = F[k * 3 + 1]; out.data[k * 4 + 2] = F[k * 3 + 2]; }
    return out;
  }

  // sRGB ↔ CIELAB (D65)
  const lin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const gam = (c) => 255 * (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
  const LUT = new Float32Array(256); for (let i = 0; i < 256; i++) LUT[i] = lin(i);
  const fL = (t) => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
  function rgb2lab(r, g, b, out) {
    const R = LUT[r | 0], G = LUT[g | 0], B = LUT[b | 0];
    const X = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047, Y = 0.2126 * R + 0.7152 * G + 0.0722 * B, Z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
    const fx = fL(X), fy = fL(Y), fz = fL(Z);
    out[0] = 116 * fy - 16; out[1] = 500 * (fx - fy); out[2] = 200 * (fy - fz); return out;
  }
  function lab2rgb(L, a, b, out) {
    const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    const inv = (t) => t ** 3 > 0.008856 ? t ** 3 : (t - 16 / 116) / 7.787;
    const X = inv(fx) * 0.95047, Y = inv(fy), Z = inv(fz) * 1.08883;
    out[0] = clamp(gam(3.2406 * X - 1.5372 * Y - 0.4986 * Z), 0, 255);
    out[1] = clamp(gam(-0.9689 * X + 1.8758 * Y + 0.0415 * Z), 0, 255);
    out[2] = clamp(gam(0.0557 * X - 0.2040 * Y + 1.0570 * Z), 0, 255);
    return out;
  }
  function hexLab(hex) { const v = parseInt(hex.slice(1), 16); return rgb2lab((v >> 16) & 255, (v >> 8) & 255, v & 255, [0, 0, 0]); }

  function labStats(img, mask) {
    const s = [0, 0, 0], s2 = [0, 0, 0], lab = [0, 0, 0]; let n = 0;
    for (let k = 0; k < img.w * img.h; k++) {
      if (!mask[k]) continue;
      rgb2lab(img.data[k * 4], img.data[k * 4 + 1], img.data[k * 4 + 2], lab);
      for (let c = 0; c < 3; c++) { s[c] += lab[c]; s2[c] += lab[c] * lab[c]; } n++;
    }
    if (!n) return null;
    return { mean: s.map((v) => v / n), std: s2.map((v, c) => Math.sqrt(Math.max(0, v / n - (s[c] / n) ** 2))), n };
  }

  function erode(m, w, h, r) {
    let cur = m;
    for (let k = 0; k < r; k++) {
      const out = new Uint8Array(cur);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x; if (!cur[i]) continue;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1 || !cur[i - 1] || !cur[i + 1] || !cur[i - w] || !cur[i + w]) out[i] = 0;
      }
      cur = out;
    }
    return cur;
  }
  function dilate(m, w, h, r) {
    let cur = m;
    for (let k = 0; k < r; k++) {
      const out = new Uint8Array(cur);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x; if (cur[i]) continue;
        if ((x > 0 && cur[i - 1]) || (x < w - 1 && cur[i + 1]) || (y > 0 && cur[i - w]) || (y < h - 1 && cur[i + w])) out[i] = 1;
      }
      cur = out;
    }
    return cur;
  }

  // Tooth outline polygon inside box b (x0..x1, y0 = gingival zenith, y1 = incisal edge)
  function toothPolygon(shape, b, canine, n = 72) {
    const w = b.x1 - b.x0, h = b.y1 - b.y0, cx = (b.x0 + b.x1) / 2;
    const P = { square: { top: 0.84, ex: 5.0, corner: 0.10 }, oval: { top: 0.76, ex: 2.6, corner: 0.30 }, triangular: { top: 0.60, ex: 3.6, corner: 0.14 } }[shape] || { top: 0.84, ex: 5, corner: 0.1 };
    const pts = [];
    for (let i = 0; i < n; i++) {
      const th = (i / n) * Math.PI * 2;                              // 0 → right middle, π/2 → bottom
      const c = Math.cos(th), s = Math.sin(th);
      let x = Math.sign(c) * Math.abs(c) ** (2 / P.ex), y = Math.sign(s) * Math.abs(s) ** (2 / P.ex);
      const v = (y + 1) / 2;                                          // 0 top … 1 incisal
      x *= P.top + (1 - P.top) * Math.min(1, v * 1.6) ** 0.8;        // narrower at the neck
      if (v < 0.5) { const zen = 1 - (0.5 - v) * 0.12; y = (v * 2 - 1) * zen; }   // softer cervical arc
      if (canine && v > 0.62) { const k = ((v - 0.62) / 0.38); x *= 1 - 0.62 * k ** 1.4; y += 0.06 * (1 - Math.abs(x)); }
      pts.push({ x: cx + x * w / 2, y: b.y0 + (y + 1) / 2 * h });
    }
    return pts;
  }

  function rasterPoly(poly, w, h) {
    const m = new Uint8Array(w * h);
    let ya = Infinity, yb = -Infinity; for (const p of poly) { ya = Math.min(ya, p.y); yb = Math.max(yb, p.y); }
    for (let y = Math.max(0, Math.floor(ya)); y <= Math.min(h - 1, Math.ceil(yb)); y++) {
      const xs = [], yy = y + 0.5;
      for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const a = poly[i], b = poly[j];
        if ((a.y > yy) !== (b.y > yy)) xs.push((b.x - a.x) * (yy - a.y) / (b.y - a.y) + a.x);
      }
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) for (let x = Math.max(0, Math.ceil(xs[k] - 0.5)); x <= Math.min(w - 1, Math.floor(xs[k + 1] - 0.5)); x++) m[y * w + x] = 1;
    }
    return m;
  }

  // Source texture for a tooth: its own pixels (or the mirror tooth's, or a library tooth), with the non-enamel part of the box filled from the enamel.
  function toothPatch(img, mask, box) {
    const x0 = Math.max(0, Math.floor(box.x0)), y0 = Math.max(0, Math.floor(box.y0)), x1 = Math.min(img.w - 1, Math.ceil(box.x1)), y1 = Math.min(img.h - 1, Math.ceil(box.y1));
    const w = Math.max(2, x1 - x0 + 1), h = Math.max(2, y1 - y0 + 1), p = makeImg(w, h), hole = new Uint8Array(w * h), inner = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const k = (y0 + y) * img.w + (x0 + x), o = (y * w + x) * 4;
      for (let c = 0; c < 4; c++) p.data[o + c] = img.data[k * 4 + c];
      if (mask[k]) inner[y * w + x] = 1; else hole[y * w + x] = 1;
    }
    const core = erode(inner, w, h, 1);                    // avoid edge pixels mixed with gum / gaps
    const fillMask = new Uint8Array(w * h); let any = 0;
    for (let k = 0; k < w * h; k++) { fillMask[k] = core[k] ? 0 : 1; any += core[k]; }
    const filled = any > 10 ? inpaint(p, fillMask) : p;
    return { img: filled, mask: core, stats: labStats(filled, core) };
  }

  /*
    opts: {
      crop: original mouth crop image, mouthMask: Uint8 (inner-lip polygon, crop size),
      toothMasks: {t: Uint8 mask (crop size)} of the patient's current teeth (selected upper six),
      current: {t: box}, target: {t: box}, treat: {t: 'veneer'|'crown'|'composite'|'implant'|'pontic'|'none'...},
      status: {t: 'present'|'missing'|'broken'|'hidden'|'unsure'}, shape, shadeHex, whiten (0..1), library: {t: {img, mask}} optional
    }
    returns { img, alpha } — the edited crop and the blend weight for pasting back
  */
  function simulate(o) {
    const { crop, mouthMask } = o, w = crop.w, h = crop.h;
    const treated = TEETH.filter((t) => o.target[t] && (o.treat[t] || 'veneer') !== 'none');
    // 1. remove the treated teeth (fill from their surroundings) so narrower / shorter new teeth reveal mouth, not old enamel
    const old = new Uint8Array(w * h);
    for (const t of treated) { const m = o.toothMasks[t]; if (m) for (let k = 0; k < w * h; k++) if (m[k]) old[k] = 1; }
    const oldD = dilate(old, w, h, 3);
    const hole = new Uint8Array(w * h); for (let k = 0; k < w * h; k++) hole[k] = oldD[k] && mouthMask[k] ? 1 : 0;
    const base = inpaint(crop, hole);
    // the space behind the teeth is the dark mouth interior; near the gum line it stays gum-coloured (papillae)
    const cur = TEETH.map((t) => o.current[t]).filter(Boolean);
    const gumY = cur.length ? cur.reduce((a, b) => a + b.y0, 0) / cur.length : 0, tH = cur.length ? cur.reduce((a, b) => a + (b.y1 - b.y0), 0) / cur.length : h;
    const dark = darkColour(crop, mouthMask, old);
    for (let y = 0; y < h; y++) {
      const g = clamp((y - gumY - 0.12 * tH) / (0.3 * tH), 0, 1) * 0.92;
      if (g <= 0) continue;
      for (let x = 0; x < w; x++) { const i = y * w + x; if (hole[i]) for (let c = 0; c < 3; c++) base.data[i * 4 + c] = base.data[i * 4 + c] * (1 - g) + dark[c] * g; }
    }
    // 2. reference enamel statistics and target colour
    const allTeeth = new Uint8Array(w * h);
    for (const t of TEETH) { const m = o.toothMasks[t]; if (m && o.status[t] !== 'missing') for (let k = 0; k < w * h; k++) if (m[k]) allTeeth[k] = 1; }
    const ref = labStats(crop, erode(allTeeth, w, h, 2)) || { mean: [75, 2, 15], std: [8, 2, 4] };
    const shade = hexLab(o.shadeHex || '#EEE6D4'), s = clamp(o.whiten ?? 1, 0, 1);
    const tgtL = ref.mean[0] + (Math.max(shade[0], ref.mean[0]) - ref.mean[0]) * s;
    const tgtA = ref.mean[1] + (shade[1] - ref.mean[1]) * s, tgtB = ref.mean[2] + (shade[2] - ref.mean[2]) * s;
    // 3. draw new teeth: canines first, centrals last (they overlap the laterals)
    const out = copyImg(base), lab = [0, 0, 0], rgb = [0, 0, 0], px = [0, 0, 0];
    const order = ['13', '23', '12', '22', '11', '21'].filter((t) => treated.includes(t));
    const patches = {};
    for (const t of order) {
      const usable = (k) => o.status[k] !== 'missing' && o.status[k] !== 'broken' && o.status[k] !== 'hidden' && o.toothMasks[k] && o.current[k];
      let srcKey = null, flip = false, patch = null;
      if (usable(t)) srcKey = t; else if (usable(MIRROR[t])) { srcKey = MIRROR[t]; flip = true; }
      if (srcKey) patch = patches[srcKey] || (patches[srcKey] = toothPatch(crop, o.toothMasks[srcKey], o.current[srcKey]));
      else if (o.library && o.library[t]) patch = o.library[t];
      else if (o.library && o.library[MIRROR[t]]) { patch = o.library[MIRROR[t]]; flip = true; }
      if (!patch || !patch.stats) continue;
      const st = patch.stats, kL = clamp(6 / Math.max(1, st.std[0]), 0.55, 1.0);   // keep enamel texture, soften stains
      const b = o.target[t], poly = toothPolygon(o.shape || 'square', b, t === '13' || t === '23');
      const pm = rasterPoly(poly, w, h), alpha = blurAlpha(Float32Array.from(pm), w, h, 1);
      const bx0 = Math.max(0, Math.floor(b.x0) - 2), bx1 = Math.min(w - 1, Math.ceil(b.x1) + 2), by0 = Math.max(0, Math.floor(b.y0) - 2), by1 = Math.min(h - 1, Math.ceil(b.y1) + 2);
      const bw = b.x1 - b.x0, bh = b.y1 - b.y0, P = patch.img;
      for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) {
        const k = y * w + x, a = Math.min(alpha[k] * 1.15, 1) * (mouthMask[k] ? 1 : 0); if (a <= 0.01) continue;
        const u = clamp((x - b.x0) / bw, 0, 1), v = clamp((y - b.y0) / bh, 0, 1);
        sample(P, (flip ? 1 - u : u) * (P.w - 1), v * (P.h - 1), px);
        rgb2lab(px[0], px[1], px[2], lab);
        let L = tgtL + (lab[0] - st.mean[0]) * kL, A = tgtA + (lab[1] - st.mean[1]) * 0.5, B = tgtB + (lab[2] - st.mean[2]) * 0.5;
        // form: rounded labial surface (darker towards contacts), cervical shadow, incisal translucency
        const side = Math.abs(2 * u - 1);
        L -= 9 * side ** 3 + 5 * Math.max(0, 0.18 - v) / 0.18;
        if (v > 0.84) { const q = (v - 0.84) / 0.16; L -= 6 * q; B -= 5 * q; A -= 0.5 * q; }
        lab2rgb(L, A, B, rgb);
        for (let c = 0; c < 3; c++) out.data[k * 4 + c] = out.data[k * 4 + c] * (1 - a) + rgb[c] * a;
      }
    }
    // 4. untreated teeth keep their original pixels; lips stay on top (only the mouth interior changes)
    for (const t of TEETH) if (!treated.includes(t) && o.toothMasks[t]) { const m = o.toothMasks[t]; for (let k = 0; k < w * h; k++) if (m[k] && !hole[k]) for (let c = 0; c < 3; c++) out.data[k * 4 + c] = crop.data[k * 4 + c]; }
    const alpha = blurAlpha(Float32Array.from(mouthMask), w, h, 1);
    for (let k = 0; k < w * h; k++) {
      const a = alpha[k]; if (a >= 1) continue;
      for (let c = 0; c < 3; c++) out.data[k * 4 + c] = crop.data[k * 4 + c] * (1 - a) + out.data[k * 4 + c] * a;
    }
    return { img: out, alpha };
  }

  // typical colour of the dark mouth interior: the darkest fifth of the mouth pixels that are not teeth
  function darkColour(img, mouth, teeth) {
    const px = [];
    for (let k = 0; k < img.w * img.h; k += 2) if (mouth[k] && !teeth[k]) { const i = k * 4; px.push([0.299 * img.data[i] + 0.587 * img.data[i + 1] + 0.114 * img.data[i + 2], i]); }
    if (px.length < 20) return [35, 18, 20];
    px.sort((a, b) => a[0] - b[0]);
    const n = Math.max(5, Math.floor(px.length * 0.2)), s = [0, 0, 0];
    for (let j = 0; j < n; j++) for (let c = 0; c < 3; c++) s[c] += img.data[px[j][1] + c];
    return s.map((v) => v / n);
  }

  const api = { TEETH, makeImg, copyImg, sample, warpAffine, pasteBack, blurAlpha, inpaint, rgb2lab, lab2rgb, hexLab, labStats, erode, dilate, toothPolygon, rasterPoly, toothPatch, simulate };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.DSDSim = api;
})(typeof window !== 'undefined' ? window : globalThis);
