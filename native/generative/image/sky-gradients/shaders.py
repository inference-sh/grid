"""GLSL for the sky renderer, from the sky gradients page (WebGL2 -> GLSL 330).

Changes from the page: uOffset for tiled renders, aurora and light leak take an
explicit time so a video can crossfade its tail into its head (uLoopLen, uFade),
and uGrainT reseeds the film grain per frame when grain is animated.
"""

VERT = """#version 330
void main(){ vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }
"""

FRAG = """#version 330
out vec4 fragColor;
uniform sampler2D uGrid;
uniform ivec2 uGridSize;
uniform vec2 uRes, uOffset;
uniform float uAz0, uAz1, uElBot, uElTop, uCurve;
uniform vec3 uSunCol;
uniform float uSunEl, uSunSize, uBloom, uShowSun, uPxPerDeg;
uniform vec2 uSunPx;
uniform float uAurora, uAuroraEl, uAuroraRed, uDip, uMirror, uTime, uAurPal, uAurHue, uAurVar, uAurSpike, uLeakMove, uAurMove;
uniform float uExposure, uGrain, uGrainSize, uLeak, uLeakAngle, uVignette, uSat, uWarmth, uContrast, uLift, uSeed;
uniform float uLoopLen, uFade, uGrainT, uTimeBase;
const vec3 INK = vec3(0.102, 0.082, 0.114);

float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y); }
vec4 fetchG(ivec2 p){ return texelFetch(uGrid, clamp(p, ivec2(0), uGridSize - 1), 0); }
vec4 crom(vec4 a, vec4 b, vec4 c, vec4 d, float t){ return b + 0.5 * t * (c - a + t * (2.0 * a - 5.0 * b + 4.0 * c - d + t * (3.0 * (b - c) + d - a))); }
vec4 bicubic(vec2 uv){
  vec2 p = uv * vec2(uGridSize) - 0.5;
  ivec2 i = ivec2(floor(p)); vec2 f = p - floor(p);
  vec4 r0 = crom(fetchG(i + ivec2(-1, -1)), fetchG(i + ivec2(0, -1)), fetchG(i + ivec2(1, -1)), fetchG(i + ivec2(2, -1)), f.x);
  vec4 r1 = crom(fetchG(i + ivec2(-1, 0)), fetchG(i + ivec2(0, 0)), fetchG(i + ivec2(1, 0)), fetchG(i + ivec2(2, 0)), f.x);
  vec4 r2 = crom(fetchG(i + ivec2(-1, 1)), fetchG(i + ivec2(0, 1)), fetchG(i + ivec2(1, 1)), fetchG(i + ivec2(2, 1)), f.x);
  vec4 r3 = crom(fetchG(i + ivec2(-1, 2)), fetchG(i + ivec2(0, 2)), fetchG(i + ivec2(1, 2)), fetchG(i + ivec2(2, 2)), f.x);
  return max(crom(r0, r1, r2, r3, f.y), vec4(0.0));
}
const vec3 PAL[24] = vec3[24](
  vec3(0.05, 1.0, 0.45), vec3(0.45, 1.0, 0.1), vec3(1.0, 0.22, 0.38),
  vec3(1.0, 0.35, 0.6), vec3(1.0, 0.5, 0.58), vec3(0.75, 0.35, 0.95),
  vec3(1.0, 0.55, 0.4), vec3(1.0, 0.68, 0.5), vec3(1.0, 0.4, 0.6),
  vec3(0.7, 0.5, 1.0), vec3(0.88, 0.58, 0.95), vec3(0.55, 0.4, 1.0),
  vec3(0.88, 0.35, 0.85), vec3(1.0, 0.45, 0.7), vec3(0.6, 0.3, 0.95),
  vec3(1.0, 0.42, 0.42), vec3(1.0, 0.6, 0.45), vec3(0.9, 0.3, 0.6),
  vec3(0.62, 0.3, 0.72), vec3(0.82, 0.42, 0.75), vec3(0.45, 0.3, 0.85),
  vec3(1.0, 0.6, 0.45), vec3(1.0, 0.35, 0.65), vec3(0.6, 0.35, 1.0));
vec3 hueRot(vec3 c, float h){
  const vec3 k = vec3(0.57735);
  float ch = cos(h);
  return c * ch + cross(k, c) * sin(h) + k * dot(k, c) * (1.0 - ch);
}
float ribbonD(float a, float D0, float seed, float t){
  return D0 * (1.0 + 0.32 * sin(a * 2.2 + seed + t * 0.07) + 0.14 * sin(a * 5.3 - seed * 1.7 + t * 0.13) + 0.22 * (vnoise(vec2(a * 3.0 + seed, t * 0.05)) - 0.5));
}
vec3 sheet(float a, float el, float D0, float dMul, float path, float seed, float t, float red, int pal, float var, float spike){
  float d = max(ribbonD(a, D0, path, t), 40.0) * dMul;
  float d2 = max(ribbonD(a + 0.01, D0, path, t), 40.0) * dMul;
  float slope = (d2 - d) / (0.01 * d);
  float fold = clamp(sqrt(1.0 + slope * slope), 1.0, 2.4);
  float drop = d * d / (2.0 * 6360.0);
  float eb = degrees(atan((100.0 - drop) / d));
  float et = degrees(atan((260.0 - drop) / d));
  float y = (el - eb) / max(et - eb, 0.4);
  if (y < -0.9 || y > 1.6) return vec3(0.0);
  float lift = smoothstep(2.0, 6.0, eb);
  float near = clamp(D0 / d, 0.35, 1.8);
  float soft = (0.05 + 0.06 * near) * mix(2.2, 1.0, (fold - 1.0) / 1.4);
  float base = smoothstep(-soft, soft * 0.7, y);
  float smudge = exp(min(y, 0.0) / (soft * 2.0)) * 0.25;
  float pk = vnoise(vec2(a * d * 0.012 + seed, t * 0.08)) * 0.65 + vnoise(vec2(a * d * 0.035 + seed * 2.0, t * 0.12)) * 0.35;
  float top = mix(0.35, 1.25, smoothstep(0.15, 0.85, pk));
  top *= mix(1.0, 0.72 + 0.56 * vnoise(vec2(a * d * 0.16 + seed * 5.0, t * 0.25)), spike);
  float taper = pow(clamp(1.0 - max(y, 0.0) / top, 0.0, 1.0), 1.25);
  float rays = mix(0.86, 1.1, vnoise(vec2(a * d * 0.06, t * 0.3 + seed)));
  float fine = mix(1.0, mix(0.55, 1.35, vnoise(vec2(a * d * 0.22, t * 0.45 + seed * 3.0))), spike * smoothstep(0.05, 0.35, y));
  float prof = (base * taper * fine * mix(1.0, rays, smoothstep(0.1, 0.5, y)) + smudge * (1.0 - base)) * fold * near * lift;
  float cv = vnoise(vec2(a * 2.2 + seed * 3.0, t * 0.04));
  cv = clamp(0.5 + (cv - 0.5) * (0.6 + var * 1.6), 0.0, 1.0);
  vec3 A = PAL[pal * 3], B = PAL[pal * 3 + 1], U = PAL[pal * 3 + 2];
  vec3 lowc = mix(A, B, smoothstep(0.1, 0.9, cv));
  vec3 upper = mix(U, mix(U, B, 0.5), vnoise(vec2(a * 3.0 + seed, 2.0)) * var * 0.6);
  vec3 col = mix(lowc, upper, smoothstep(0.3, 0.95, y) * clamp(0.3 + red * 0.6, 0.0, 1.0));
  float hz = (y + 0.01) / 0.06;
  return col * prof + vec3(1.0, 0.3, 0.65) * exp(-hz * hz) * 0.06 * near * lift;
}
vec3 ribbon(float a, float el, float D0, float path, float seed, float t, float red, int pal, float var, float spike){
  return sheet(a, el, D0, 1.0, path, seed, t, red, pal, var, spike) + sheet(a + 0.004, el, D0, 1.07, path, seed, t, red, pal, var, spike) * 0.55;
}
vec3 auroraField(float az, float el, float seed, float elA, float red, float t, int pal, float var, float hueShift, float spike){
  float a = radians(az);
  float D0 = 100.0 / tan(radians(clamp(elA + 8.0, 3.0, 60.0)));
  float env = 0.35 + 0.65 * smoothstep(0.15, 0.75, vnoise(vec2(a * 1.4 + seed * 0.7 + t * 0.02, 7.0 + t * 0.05)));
  vec3 c = ribbon(a, el, D0, seed, seed, t, red, pal, var, spike);
  c += ribbon(a + 0.05, el, D0 * 1.25, seed, seed + 3.1, t, red, pal, var, spike) * 0.8;
  c += ribbon(a + 0.1, el, D0 * 1.55, seed, seed + 7.3, t, red * 0.6, pal, var, spike) * 0.65;
  float gz = (el - elA - 9.0) / 12.0;
  c += PAL[pal * 3] * exp(-gz * gz) * 0.06;
  return max(hueRot(c * env, hueShift), 0.0);
}
vec3 auroraAt(float az, float el, float t){ return auroraField(az, el, uSeed, uAuroraEl, uAuroraRed, t, int(uAurPal + 0.5), uAurVar, radians(uAurHue), uAurSpike); }
vec3 leakAt(vec2 uv, float lt){
  float la = uLeakAngle + sin(lt * 0.11 + uSeed) * 0.18;
  vec2 ld = vec2(cos(la), sin(la));
  vec2 p = (uv - 0.5) * vec2(uRes.x / uRes.y, 1.0);
  float edge = dot(p, ld) / (0.5 * (abs(ld.x) * uRes.x / uRes.y + abs(ld.y)));
  float lk = smoothstep(-0.3, 1.0, edge);
  float along = dot(p, vec2(-ld.y, ld.x));
  float c1 = (hash12(vec2(uSeed, 1.7)) - 0.5) * 0.9 + sin(lt * 0.23) * 0.18, c2 = (hash12(vec2(uSeed, 8.3)) - 0.5) * 1.1 + sin(lt * 0.31 + 1.0) * 0.22;
  float b1 = exp(-pow((along - c1) / 0.3, 2.0)), b2 = exp(-pow((along - c2) / 0.16, 2.0));
  float soft = 0.75 + 0.5 * vnoise(vec2(along * 3.0 + uSeed, edge * 2.0 + lt * 0.12));
  float breathe = 0.82 + 0.18 * sin(lt * 0.45 + uSeed);
  return 2.0 * soft * breathe * (
    pow(lk, 2.4) * (0.35 + 0.9 * b1) * vec3(1.0, 0.3, 0.1) +
    pow(lk, 4.5) * b2 * vec3(1.0, 0.72, 0.32) * 1.4 +
    pow(lk, 6.0) * (1.0 - b1) * vec3(1.0, 0.2, 0.5) * 0.6);
}
// weight of the previous loop: 0 except over the last uFade seconds, where the tail fades into the head
float loopW(){ return uLoopLen > 0.0 ? smoothstep(uLoopLen - uFade, uLoopLen, uTime) : 0.0; }
vec3 aces(vec3 x){ return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }

void main(){
  vec2 frag = gl_FragCoord.xy + uOffset;
  vec2 uv = frag / uRes;
  vec4 g = bicubic(uv);
  vec3 col = g.rgb;
  float okA = clamp(g.a, 0.0, 1.0);
  if (uShowSun > 0.5) {
    float ang = radians(length(frag - uSunPx) / uPxPerDeg);
    float rs = 0.0075 * uSunSize;
    float x = clamp(ang / rs, 0.0, 1.0);
    float limb = 1.0 - 0.5 * (1.0 - sqrt(max(1.0 - x * x, 0.0)));
    col += uSunCol * (smoothstep(1.06, 0.92, ang / rs) * limb * 60.0 + uBloom * (0.5 * exp(-ang / (rs * 2.2)) + 0.12 * exp(-ang * 9.0)));
  }
  vec3 c = col * uExposure;
  float w = loopW();
  if (uAurora > 0.0) {
    float el = uElBot + (uElTop - uElBot) * pow(uv.y, uCurve);
    float az = mix(uAz0, uAz1, uv.x);
    float gain = 1.0;
    if (el < uDip) { if (uMirror > 0.5) { el = 2.0 * uDip - el; gain = 0.55; } else gain = 0.0; }
    if (gain > 0.0) {
      vec3 au = auroraAt(az, el, (uTimeBase + uTime) * uAurMove);
      if (w > 0.0 && uAurMove > 0.0) au = mix(au, auroraAt(az, el, uTimeBase + uTime - uLoopLen), w);
      float dark = mix(0.3, 1.0, smoothstep(-1.0, -9.0, uSunEl));
      c += au * uAurora * 0.2 * dark * gain;
    }
  }
  if (uLeak > 0.0) {
    vec3 lc = leakAt(uv, (uTimeBase + uTime) * uLeakMove);
    if (w > 0.0 && uLeakMove > 0.0) lc = mix(lc, leakAt(uv, uTimeBase + uTime - uLoopLen), w);
    c += uLeak * lc;
  }
  c *= vec3(1.0 + uWarmth * 0.14, 1.0 + uWarmth * 0.02, 1.0 - uWarmth * 0.14);
  c = aces(c);
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = max(mix(vec3(l), c, uSat), 0.0);
  c = pow(c, vec3(1.0 / 2.2));
  c = clamp((c - 0.5) * uContrast + 0.5, 0.0, 1.0);
  float vig = smoothstep(1.25, 0.25, length((uv - 0.5) * vec2(uRes.x / uRes.y, 1.0) * 0.9));
  c *= mix(1.0, vig, uVignette);
  c = INK * uLift + c * (1.0 - INK * uLift);
  c = mix(INK, c, okA);
  vec2 gc = floor(frag / uGrainSize);
  float gt = fract(uGrainT * 0.618034) * 517.0;
  float gr = hash12(gc + fract(uSeed * 0.137) * 731.0 + gt) + hash12(gc * 1.37 + 91.7 + fract(uSeed * 0.71) * 311.0 + gt) - 1.0;
  c += gr * uGrain * 0.11 * (0.35 + 0.65 * (1.0 - l));
  c += (hash12(frag + 3.7) + hash12(frag * 1.71 + 9.2) - 1.0) / 255.0;
  fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
"""

# Atmosphere ray-march (port of sky() / computeGrid() in sky.py). uMode 0 renders the 12
# ambient sample directions into a 12x1 target; uMode 1 renders one ray per grid cell.
SKY_FRAG = """#version 330
out vec4 o;
uniform sampler2D uLut;
uniform int uMode, uN, uBelow;
uniform ivec2 uGridSize;
uniform float uAlt, uMieG, uMs, uMieS, uMieE;
uniform vec3 uBetaR, uOzA, uStratS, uStratE, uSmokeE, uSmokeS, uSd, uAmb;
uniform float uLook, uFov, uElBot, uElTop, uCurve, uDip;
const float R = 6360.0, PI = 3.14159265359;
const int LUT_W = 128, LUT_H = 40;

vec3 lutAt(int x, int y){ return texelFetch(uLut, ivec2(x, y), 0).rgb; }
vec3 lutSample(vec3 P, vec3 sd){
  float r = length(P), h = r - R;
  float e = asin(clamp(dot(P, sd) / r, -1.0, 1.0));
  float v = sign(e) * sqrt(abs(e) / (PI / 2.0));
  float x = clamp((v + 0.37) / 1.37, 0.0, 1.0) * float(LUT_W - 1), y = sqrt(clamp(h / 100.0, 0.0, 1.0)) * float(LUT_H - 1);
  int x0 = min(int(floor(x)), LUT_W - 2), y0 = min(int(floor(y)), LUT_H - 2);
  float fx = x - float(x0), fy = y - float(y0);
  return mix(mix(lutAt(x0, y0), lutAt(x0 + 1, y0), fx), mix(lutAt(x0, y0 + 1), lutAt(x0 + 1, y0 + 1), fx), fy);
}
float hg(float c, float g){ return (1.0 - g * g) / (4.0 * PI * pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5)); }
vec3 dirFrom(float az, float el){ float a = radians(az), e = radians(el); return vec3(sin(a) * cos(e), sin(e), cos(a) * cos(e)); }
vec3 sky(vec3 rd, vec3 amb, int N){
  float r0 = R + uAlt, b = r0 * rd.y, cc = (uAlt - 100.0) * (2.0 * R + uAlt + 100.0);
  float tMax = -b + sqrt(max(b * b - cc, 0.0));
  float mu = dot(rd, uSd);
  float phR = 3.0 / (16.0 * PI) * (1.0 + mu * mu), phM = hg(mu, uMieG), phS = hg(mu, 0.72), phK = hg(mu, 0.55);
  float iso = uMs / (4.0 * PI), am = uMs * 2.2;
  vec3 L = vec3(0.0), T = vec3(1.0);
  float tp = 0.0;
  for (int i = 0; i < N; i++) {
    float s = float(i + 1) / float(N), t = tMax * s * s, dt = t - tp, tm = 0.5 * (t + tp);
    tp = t;
    vec3 P = vec3(rd.x * tm, r0 + rd.y * tm, rd.z * tm);
    float h = max(length(P) - R, 0.0);
    float zz = (h - 20.0) / 4.0;
    float dr = exp(-h / 8.0), dm = exp(-h / 1.2), dO = max(0.0, 1.0 - abs(h - 25.0) / 15.0), ds = exp(-zz * zz), dk = exp(-h / 2.5);
    vec3 TS = lutSample(P, uSd);
    vec3 sR = uBetaR * dr, sS = uStratS * ds, sK = uSmokeS * dk;
    float sM = uMieS * dm;
    vec3 ext = sR + uMieE * dm + uOzA * dO + uStratE * ds + uSmokeE * dk;
    vec3 Sv = (sR * phR + sM * phM + sS * phS + sK * phK + (sR + sM + sS + sK) * iso) * TS + (sR + sM + sS) * amb * am;
    vec3 st = exp(-ext * dt);
    L += T * Sv * (1.0 - st) / max(ext, vec3(1e-7));
    T *= st;
  }
  return L * 20.0;
}
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy);
  if (uMode == 0) {
    int k = p.x / 4;
    float el = k == 0 ? 8.0 : (k == 1 ? 35.0 : 75.0);
    o = vec4(sky(dirFrom(float(p.x - k * 4) * 90.0, el), vec3(0.0), 24), 1.0);
    return;
  }
  float el = uElBot + (uElTop - uElBot) * pow((float(p.y) + 0.5) / float(uGridSize.y), uCurve);
  float e = el, f = 1.0, a = 1.0;
  if (el < uDip + 0.05) {
    float depth = radians(uDip - el);
    if (uBelow == 1) { e = uDip + 0.05 + (uDip - el); float cosi = sin(max(depth, 0.0)); f = 0.15 + 0.85 * (0.02 + 0.98 * pow(1.0 - cosi, 5.0)); }
    else if (uBelow == 2) { e = uDip + 0.05; f = 1.0 / (1.0 + degrees(depth) * 0.12); }
    else { e = uDip + 0.05; a = 0.0; }
  }
  float az = uLook - uFov / 2.0 + uFov * (float(p.x) + 0.5) / float(uGridSize.x);
  o = vec4(sky(dirFrom(az, e), uAmb, uN) * f, a);
}
"""

# RGB frame -> packed I420 (BT.709, limited range), so readback is 1.5 bytes per pixel and
# comes out in ffmpeg's yuv420p byte order. One R8 texel per output byte: the target is
# w x (h * 3 / 2) and texel (x, y) is byte y * w + x of the I420 buffer.
YUV_FRAG = """#version 330
out vec4 o;
uniform sampler2D uFrame;
uniform ivec2 uSize;
vec3 px(int x, int y){ return texelFetch(uFrame, ivec2(x, uSize.y - 1 - y), 0).rgb; }
float luma(vec3 c){ return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
void main(){
  int w = uSize.x, h = uSize.y, cw = w / 2, ch = h / 2;
  int idx = int(gl_FragCoord.y) * w + int(gl_FragCoord.x);
  float v;
  if (idx < w * h) {
    v = 16.0 + 219.0 * luma(clamp(px(idx % w, idx / w), 0.0, 1.0));
  } else {
    int j = idx - w * h;
    bool isV = j >= cw * ch;
    if (isV) j -= cw * ch;
    int x = (j % cw) * 2, y = (j / cw) * 2;
    vec3 c = clamp((px(x, y) + px(x + 1, y) + px(x, y + 1) + px(x + 1, y + 1)) * 0.25, 0.0, 1.0);
    float Y = luma(c);
    v = isV ? 128.0 + 224.0 * (c.r - Y) / 1.5748 : 128.0 + 224.0 * (c.b - Y) / 1.8556;
  }
  o = vec4(v / 255.0, 0.0, 0.0, 1.0);
}
"""
