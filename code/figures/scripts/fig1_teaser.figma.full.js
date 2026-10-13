const page = await figma.getNodeByIdAsync('532:2'); await figma.setCurrentPageAsync(page);
const FR = await figma.getNodeByIdAsync('544:580'); for (const k of [...FR.children]) k.remove(); FR.resize(520, 480);
// figma_helpers.js — house drawing helpers for use_figma scripts (ARR-Oct, FIGURE_STANDARD.md).
// Paste this block at the top of every use_figma script, then set FR to your target frame.
// Source of every number: FIGURE_STANDARD §2–§3 (1080-px figure* / 520-px column canvases,
// Inter 19.1 px labels, STIX Two Text 20 px math with 15.5 px sub/superscripts, pills #2F6B75,
// panels r=14 + 2-px mid-tone stroke, ink #1A1A1A, separate filled arrowheads).
// Used for P1 Fig. 2 (544:552) and P3 Fig. 2 (544:584); read those frames as worked examples.
//
// Rules the helpers encode (do not work around them):
//  * Text sits on a BASELINE (y you pass = baseline). Inter and STIX align because K[] holds the
//    font's baseline offset for lineHeight = 1.25 × size (measured from the fonts' hhea metrics).
//  * Figma trims trailing spaces in auto-width text: RUN() adds the space width itself, so write
//    runs as ['per checkpoint ', F.R], ['θ', F.sBI], ['t', F.sI, SS, INK, SUB] and never pad by hand.
//  * Math: variables Italic (F.sI), operators/digits/roman subscripts Regular (F.sR), vectors Bold
//    Italic (F.sBI) — must match <paper>/paper/macros.tex. Subscript = SS size with dy = SUB.
//  * Arrow heads are separate filled triangles (ARROW does it). halo:true draws a white underlay so a
//    crossing line reads as passing under.
//  * Colours only from the ARR palette; roles per paper in FIGURE_STANDARD §5.3.
//  * After export: figures/tools/finalize_figma_pdf.sh <raw.pdf> <paper>/figures/export/<name>.pdf
//    (Figma PDFs contain Type 3 fonts; the script outlines text and checks the result).
//
// Minimal skeleton:
//   const page = await figma.getNodeByIdAsync('532:2'); await figma.setCurrentPageAsync(page);
//   const FR = await figma.getNodeByIdAsync('<frame id>'); for (const k of [...FR.children]) k.remove();
//   ...paste helpers...; await loadFonts();
//   PANEL(6, 18, 256, 352, 'grey'); PILL(16, 2, '① Items'); T('label', 18, 64); ARROW([[a,b],[c,d]]);
//   FR.exportSettings = [{ format: 'PDF', contentsOnly: true, suffix: '', colorProfile: 'DOCUMENT' }];
//   return { frame: FR.id, texts: FR.findAll(n => n.type === 'TEXT').length };

const F = { R: { family: 'Inter', style: 'Regular' }, B: { family: 'Inter', style: 'Bold' }, I: { family: 'Inter', style: 'Italic' }, SB: { family: 'Inter', style: 'Semi Bold' },
  sR: { family: 'STIX Two Text', style: 'Regular' }, sI: { family: 'STIX Two Text', style: 'Italic' }, sB: { family: 'STIX Two Text', style: 'Bold' },
  sBI: { family: 'STIX Two Text', style: 'Bold Italic' }, m: { family: 'STIX Two Math', style: 'Regular' } };
async function loadFonts() { for (const f of Object.values(F)) await figma.loadFontAsync(f); }
const rgb = h => ({ r: parseInt(h.slice(1, 3), 16) / 255, g: parseInt(h.slice(3, 5), 16) / 255, b: parseInt(h.slice(5, 7), 16) / 255 });
const solid = (h, o) => (o === undefined ? { type: 'SOLID', color: rgb(h) } : { type: 'SOLID', color: rgb(h), opacity: o });
// ARR palette (FIGURE_STANDARD §5.1) + tints (§5.2)
const INK = '#1A1A1A', INK2 = '#333333', MUT = '#555555', GREY = '#8C8C8C', BLUE = '#0072B2', ORANGE = '#E69F00', GREEN = '#009E73',
  VERM = '#D55E00', PURP = '#CC79A7', SKY = '#56B4E9', PILLC = '#2F6B75';
const TINT = { grey: ['#F2F2F2', '#A3A3A3'], blue: ['#EAF2F8', '#86A9C6'], orange: ['#FFF6E5', '#D4AD62'], green: ['#E6F2EC', '#86C3A5'],
  purp: ['#F7EEF4', '#C98FB2'], verm: ['#FBE9E1', '#E3A585'], violet: ['#EEEAF7', '#A393CC'] };
const K = { 'Inter': 0.9888, 'STIX Two Text': 0.887, 'STIX Two Math': 0.887 };   // baseline offset / size at lineHeight 1.25
const SPC = { 'Inter': 0.28, 'STIX Two Text': 0.25, 'STIX Two Math': 0.25 };      // space width / size
const SUB = 5, SUP = -8, SS = 15.5;                                              // subscript dy, superscript dy, sub size

// text on a baseline; align 'L' | 'C' | 'R'
function T(s, x, base, o = {}) {
  const { font = F.R, size = 19.1, color = INK, align = 'L', name } = o;
  const t = figma.createText(); t.fontName = font; t.fontSize = size; t.lineHeight = { unit: 'PIXELS', value: Math.round(size * 12.5) / 10 };
  t.characters = s; t.fills = [solid(color)]; t.textAutoResize = 'WIDTH_AND_HEIGHT'; FR.appendChild(t); t.y = base - K[font.family] * size;
  t.x = align === 'L' ? x : align === 'C' ? x - t.width / 2 : x - t.width; if (name) t.name = name; return t;
}
const adv = (s, font, size, t) => t.width + (s.endsWith(' ') ? SPC[font.family] * size : 0);
// inline runs of mixed fonts: [text, font, size?, color?, dy?]; returns { w, x0 }; name groups them ('math:...')
function RUN(runs, x, base, o = {}) {
  const { align = 'L', name } = o; const nodes = []; let cx = x;
  for (const r of runs) { const [s, font, size = (font.family === 'Inter' ? 19.1 : 20), color = o.color || INK, dy = 0] = r;
    const lead = s.startsWith(' ') ? SPC[font.family] * size : 0; const t = T(s.trim(), cx + lead, base + dy, { font, size, color }); nodes.push(t); cx += lead + adv(s, font, size, t); }
  const w = cx - x, shift = align === 'C' ? -w / 2 : align === 'R' ? -w : 0; if (shift) for (const n of nodes) n.x += shift;
  if (name && nodes.length > 1) { const g = figma.group(nodes, FR); g.name = name; } return { w, x0: x + shift };
}
function runW(runs) { let w = 0; for (const r of runs) { const size = r[2] || (r[1].family === 'Inter' ? 19.1 : 20); const s = r[0];
  const t = figma.createText(); t.fontName = r[1]; t.fontSize = size; t.characters = s.trim() || '.';
  w += (s.startsWith(' ') ? SPC[r[1].family] * size : 0) + adv(s, r[1], size, t); t.remove(); } return w; }

function RECT(x, y, w, h, o = {}) { const { fill = null, stroke = null, sw = 1.6, r = 0, dash = null, name } = o; const n = figma.createRectangle(); FR.appendChild(n); n.x = x; n.y = y; n.resize(w, h);
  n.fills = fill ? [solid(fill)] : []; n.strokes = stroke ? [solid(stroke)] : []; n.strokeWeight = sw; n.strokeAlign = 'CENTER'; if (dash) n.dashPattern = dash; n.cornerRadius = r; if (name) n.name = name; return n; }
function ELL(cx, cy, w, o = {}) { const { fill = null, stroke = null, sw = 1.2, h = w, dash = null } = o; const n = figma.createEllipse(); FR.appendChild(n); n.resize(w, h); n.x = cx - w / 2; n.y = cy - h / 2;
  n.fills = fill ? [solid(fill)] : []; n.strokes = stroke ? [solid(stroke)] : []; n.strokeWeight = sw; if (dash) n.dashPattern = dash; return n; }
// vector from a node-local path d, placed at (x0, y0)  (vectorPaths are node-local: draw from 0,0)
function PATH(d, x0, y0, o = {}) { const { stroke = INK, sw = 1.6, dash = null, fill = null, name } = o; const v = figma.createVector(); FR.appendChild(v);
  v.vectorPaths = [{ windingRule: fill ? 'NONZERO' : 'NONE', data: d }]; v.x = x0; v.y = y0; v.strokes = stroke ? [solid(stroke)] : []; v.strokeWeight = sw; v.fills = fill ? [solid(fill)] : [];
  v.strokeCap = 'NONE'; v.strokeJoin = 'ROUND'; if (dash) v.dashPattern = dash; if (name) v.name = name; return v; }
function POLY(pts, o = {}) { const x0 = Math.min(...pts.map(p => p[0])), y0 = Math.min(...pts.map(p => p[1])); return PATH(pts.map((p, i) => (i ? 'L ' : 'M ') + (p[0] - x0) + ' ' + (p[1] - y0)).join(' '), x0, y0, o); }
function HEAD(tx, ty, ang, color = INK, L = 9, W = 8) { const bx = tx - L * Math.cos(ang), by = ty - L * Math.sin(ang), px = -Math.sin(ang) * W / 2, py = Math.cos(ang) * W / 2;
  const pts = [[tx, ty], [bx + px, by + py], [bx - px, by - py]]; const x0 = Math.min(...pts.map(p => p[0])), y0 = Math.min(...pts.map(p => p[1]));
  return PATH(pts.map((p, i) => (i ? 'L ' : 'M ') + (p[0] - x0) + ' ' + (p[1] - y0)).join(' ') + ' Z', x0, y0, { stroke: null, fill: color, name: 'arrowhead' }); }
// polyline with a solid head at the last point
function ARROW(pts, o = {}) { const { color = INK, sw = 1.6, dash = null, halo = false } = o; const n = pts.length, [ax, ay] = pts[n - 2], [bx, by] = pts[n - 1], ang = Math.atan2(by - ay, bx - ax), L = 9;
  const q = pts.slice(0, n - 1).concat([[bx - (L - 1) * Math.cos(ang), by - (L - 1) * Math.sin(ang)]]); if (halo) POLY(q, { stroke: '#FFFFFF', sw: sw + 5, name: 'halo' });
  POLY(q, { stroke: color, sw, dash }); HEAD(bx, by, ang, color); }

// chassis
function PANEL(x, y, w, h, tint) { return RECT(x, y, w, h, { fill: TINT[tint][0], stroke: TINT[tint][1], sw: 2, r: 14, name: 'panel-' + tint }); }
function PILL(x, y, s) { const t = T(s, x + 14, y + 23.5, { font: F.B, size: 20.5, color: '#FFFFFF' });   // panel.x + 10, y = panel.y - 16
  const r = RECT(x, y, t.width + 28, 32, { fill: PILLC, r: 16, name: 'pill' }); FR.insertChild(FR.children.indexOf(t), r); return r; }
// rounded chip around runs (Inter or STIX)
function CHIP(runs, x, y, stroke, o = {}) { const { h = 28, sw = 1.6, pad = 6, fill = '#FFFFFF' } = o; const rr = RECT(x, y, 10, h, { fill, stroke, sw, r: 6 });
  const res = RUN(runs, x + pad, y + h / 2 + (runs[0][1].family === 'Inter' ? 6.9 : 6.6)); rr.resize(res.w + 2 * pad, h); return res.w + 2 * pad; }
// centred row of chips: items = [[runs, strokeColour, strokeWeight?], ...]
function CHIPROW(items, cx, y, o = {}) { const { gap = 5, pad = 6 } = o; const ws = items.map(it => Math.max(28, runW(it[0]) + 2 * pad)); const tot = ws.reduce((a, b) => a + b, 0) + gap * (items.length - 1);
  let x = cx - tot / 2; items.forEach((it, i) => { CHIP(it[0], x, y, it[1], { pad, sw: it[2] || 1.6 }); x += ws[i] + gap; }); return tot; }
// formula pill: white, ink stroke, r = 14; returns the rect
function FORMULA(runs, cx, yTop, o = {}) { const { h = 36 } = o; const w = runW(runs) + 20; const r = RECT(cx - w / 2, yTop, w, h, { fill: '#FFFFFF', stroke: INK, sw: 1.6, r: 14 });
  RUN(runs, cx, yTop + h / 2 + 6.6, { align: 'C', name: o.name || 'math:formula' }); return r; }

// idioms (FIGURE_STANDARD §3): frozen = LOCK, delayed/feedback = CLOCK on a dashed vermilion arrow, data = DB,
// tool = square box + gear icon, decision = gate diamond (4:197), knob = training hyper-parameter
function LOCK(x, y, color = MUT) { RECT(x, y + 6, 12, 9, { fill: color, r: 2, name: 'lock' }); PATH('M 2.5 7 L 2.5 4 C 2.5 0.3 9.5 0.3 9.5 4 L 9.5 7', x, y, { stroke: color, sw: 1.6, name: 'lock' }); }
function CLOCK(cx, cy, color = VERM) { ELL(cx, cy, 16, { fill: '#FFFFFF', stroke: color, sw: 1.6 }); POLY([[cx, cy - 5], [cx, cy], [cx + 4, cy + 2]], { stroke: color, sw: 1.4 }); }
function DB(x, y, color = INK) { PATH('M 0 4 L 0 22 C 0 27.3 22 27.3 22 22 L 22 4', x, y, { stroke: color, sw: 1.6, fill: '#FFFFFF' });
  ELL(x + 11, y + 4, 22, { h: 8, fill: '#FFFFFF', stroke: color, sw: 1.6 }); PATH('M 0 12 C 0 17.3 22 17.3 22 12', x, y, { stroke: color, sw: 1.2 }); }
function KNOB(cx, cy) { ELL(cx, cy, 22, { fill: '#FFFFFF', stroke: INK, sw: 1.4 }); POLY([[cx, cy], [cx + 6, cy - 7]], { stroke: INK, sw: 1.6 }); }
// stage block (llm-block): rounded grey block with a name and a lock (frozen) — trained variant: stroke BLUE, sw 2.6, no lock
function LLM(x, y, w, h, name, o = {}) { const { trained = false } = o;
  RECT(x, y, w, h, { fill: trained ? '#FFFFFF' : '#F7F7F7', stroke: trained ? BLUE : INK2, sw: trained ? 2.6 : 1.6, r: 8, name: 'llm-block' });
  T(name, x + w / 2 - (trained ? 0 : 8), y + h / 2 + 6.9, { font: F.SB, align: 'C' }); if (!trained) LOCK(x + w - 20, y + h / 2 - 9); }
// instance of a file component, recoloured (white fills kept). Useful component ids (FIGMA_INVENTORY §2–§3):
//   '209:11847' bot (LLM/judge) · '209:10999' magnifier (monitor) · '209:11850' pen (edit/inject) · '209:10567' check ✓
//   '209:13768' scissors (cut/delete) · '209:4267' plus (insert) · '209:11934' gear (tool) · '209:11722' brain
//   '209:11726' token · '209:10589' document · '4:197' gate diamond (decision) · '176:307' frozen MLP glyph
async function ICON(id, x, y, size, color) { const c = await figma.getNodeByIdAsync(id); const inst = c.createInstance(); FR.appendChild(inst); inst.rescale(size / Math.max(inst.width, inst.height)); inst.x = x; inst.y = y;
  for (const k of inst.findAll(() => true)) {
    if ('fills' in k && Array.isArray(k.fills) && k.fills.length) k.fills = k.fills.map(p => (p.type === 'SOLID' && !(p.color.r > 0.99 && p.color.g > 0.99 && p.color.b > 0.99)) ? { ...p, color: rgb(color) } : p);
    if ('strokes' in k && Array.isArray(k.strokes) && k.strokes.length) k.strokes = k.strokes.map(p => p.type === 'SOLID' ? { ...p, color: rgb(color) } : p); }
  return inst; }
const lerp = (a, b, t) => '#' + [0, 1, 2].map(i => Math.round(parseInt(a.slice(1 + 2 * i, 3 + 2 * i), 16) * (1 - t) + parseInt(b.slice(1 + 2 * i, 3 + 2 * i), 16) * t).toString(16).padStart(2, '0')).join('');

await loadFonts();
// ---- P3 Fig. 1 body (paste after figma_helpers.js; FR = frame 544:580). Excerpts live in X (swap and re-run to rebuild).
const X = {
  user:  { chip: 'AAE user', l1: [['… and I ', 0], ["ain't", 1], [' cooked for that many.', 0]], l2: [['My wife ', 0], ['say', 1], [' it ', 0], ['look', 1], [' easy. What should …', 0]] },
  deflt: { chip: 'default', l1: [['Hosting a dinner for ten people', 0]], l2: [['can seem daunting, but with a bit of planning …', 0]] },
  req:   { chip: 'on request', pending: true },
};
const HL = '#F9D98C';
function LINE(parts, x, base) { let cx = x; const ws = [];
  for (const [s, feat] of parts) { const w = runW([[s, F.R]]); if (feat) { const lead = s.startsWith(' ') ? SPC.Inter * 19.1 : 0; const tw = runW([[s.trim(), F.R]]); RECT(cx + lead - 2, base - 16, tw + 4, 22, { fill: HL, r: 3, name: 'feature' }); }
    RUN([[s, F.R]], cx, base); cx += w; ws.push(w); } return cx - x; }
function GCH(s, x, y, stroke, o = {}) { const w = runW([[s, F.R]]) + 12; RECT(x, y, w, 28, { fill: '#FFFFFF', stroke, sw: o.sw || 1.6, r: 6, dash: o.dash || null }); RUN([[s, F.R]], x + 6, y + 20.9); return w; }
function CARD(x, y, w, h, o = {}) { return RECT(x, y, w, h, { fill: o.fill || '#FFFFFF', stroke: o.stroke || INK, sw: o.sw || 1.6, r: 6, dash: o.dash || null, name: 'card' }); }
const out = { widths: {} };

// ===== Setting panel =====
PANEL(6, 18, 508, 260, 'grey'); PILL(16, 2, 'Setting: one AAE user');
const CX = 34, CW = 468;
// user card
CARD(CX, 40, CW, 58); let cw = CHIP([[X.user.chip, F.R]], CX + 8, 46, ORANGE, { sw: 2 });
out.widths.u1 = LINE(X.user.l1, CX + 8 + cw + 8, 67) + CX + 8 + cw + 8; out.widths.u2 = LINE(X.user.l2, CX + 8, 91);
// trunk + model
POLY([[22, 98], [22, 102]], { sw: 1.6 }); POLY([[22, 124], [22, 239]], { sw: 1.6 });
await ICON('209:11847', 12, 102, 20, MUT); T('OLMo-2-7B DPO', 40, 117, { font: F.I, color: MUT });
ARROW([[22, 161], [CX, 161]]); ARROW([[22, 239], [CX, 239]]);
// default reply card
CARD(CX, 132, CW, 58); cw = CHIP([[X.deflt.chip, F.R]], CX + 8, 138, GREY, { sw: 2 });
out.widths.d1 = LINE(X.deflt.l1, CX + 8 + cw + 8, 159) + CX + 8 + cw + 8; out.widths.d2 = LINE(X.deflt.l2, CX + 8, 183);
{ const s = 'no AAE features', w = runW([[s, F.R]]) + 12; GCH(s, CX + CW - 8 - w, 118, GREY, { dash: [4, 3] }); }
// on-request card (placeholder until exp/results/PILOT_P0/gen/L2@dpo_request.jsonl exists)
CARD(CX, 210, CW, 58, { fill: '#EDEDED', stroke: GREY, dash: [4, 3] }); cw = CHIP([[X.req.chip, F.R]], CX + 8, 216, INK2, { sw: 2 });
T('excerpt pending', CX + 8 + cw + 12, 237, { font: F.I, color: MUT, name: 'placeholder' });
{ const s = 'AAE features', w = runW([[s, F.R]]) + 12; GCH(s, CX + CW - 8 - w, 196, ORANGE, { sw: 2 }); }

// ===== Existing panel =====
const RAILY = 448, RULEY = 426;
PANEL(6, 302, 248, 172, 'verm'); PILL(16, 286, 'Existing: lost');
T('style, dialect alike', 26, 336, { font: F.I, color: MUT });
POLY([[40, RULEY], [232, RULEY]], { stroke: INK2, sw: 1, dash: [4, 3], name: 'zero-rule' }); T('0', 240, RULEY + 6.5, { color: INK });
PATH('M 0 0 C 70 -2 120 22 192 66', 42, 352, { stroke: GREY, sw: 2.4, name: 'curve-existing' });
T('converges less', 44, 408, { font: F.I, color: MUT });
POLY([[36, RAILY], [240, RAILY]], { sw: 1.6 });
[[56, 'base'], [206, 'instruct']].forEach(([x, s]) => { POLY([[x, RAILY - 5], [x, RAILY + 5]], { sw: 1.6 }); T(s, x, 468, { align: 'C' }); });

// ===== Ours panel =====
PANEL(266, 302, 248, 172, 'blue'); PILL(276, 286, 'Ours: withheld');
{ const w = runW([['can, but won’t', F.R]]) + 12 + 22; RECT(276, 312, w, 28, { fill: '#FFFFFF', stroke: BLUE, sw: 2.6, r: 6, name: 'chip-can-but-wont' }); LOCK(283, 319, MUT); RUN([['can, but won’t', F.R]], 302, 332.9); out.widths.chip = w; }
// reward-gap balance (MB_v < 0, schematic): standard reply weighs more (left pan lower)
{ const n0 = FR.children.length;
  POLY([[278, 368], [338, 354]], { sw: 2, name: 'beam' });
  POLY([[308, 361], [299, 384], [317, 384]], { sw: 1.6, fill: '#FFFFFF', name: 'fulcrum' });
  POLY([[281, 368], [281, 372]], { sw: 1 }); POLY([[335, 354], [335, 358]], { sw: 1 });
  RECT(271, 372, 22, 14, { fill: '#FFFFFF', stroke: INK, sw: 1.4, r: 2, name: 'std-card' }); POLY([[275, 377], [289, 377]], { sw: 1.2 }); POLY([[275, 381], [285, 381]], { sw: 1.2 });
  RECT(324, 358, 22, 14, { fill: '#FFFFFF', stroke: INK, sw: 1.4, r: 2, name: 'aae-card' }); POLY([[328, 363], [342, 363]], { stroke: ORANGE, sw: 2 }); POLY([[328, 367], [338, 367]], { stroke: ORANGE, sw: 2 });
  const g = figma.group(FR.children.slice(n0), FR); g.name = 'glyph-reward-gap'; }
T('reward gap', 358, 376, { color: INK });
// behavioural curves (editable group); aligned stages only, so the claim holds under C1 and under alternative delta
{ const n0 = FR.children.length;
  POLY([[410, RULEY], [496, RULEY]], { stroke: INK2, sw: 1, dash: [4, 3], name: 'zero-rule' });
  POLY([[416, 406], [476, 406]], { stroke: GREEN, sw: 2.4, name: 'style-curve' });
  POLY([[416, RULEY], [476, RULEY]], { stroke: VERM, sw: 2.4, dash: [4, 3], name: 'dialect-withheld' });
  [416, 476].forEach(x => { ELL(x, 406, 9, { fill: GREEN, stroke: INK, sw: 1 }); ELL(x, RULEY, 9, { fill: ORANGE, stroke: INK, sw: 1 }); });
  const g = figma.group(FR.children.slice(n0), FR); g.name = 'ours-curves (editable)'; }
T('style', 402, 412, { align: 'R' }); T('dialect', 402, 432, { align: 'R' }); T('0', 502, RULEY + 6.5, { color: INK });
POLY([[286, RAILY], [496, RAILY]], { sw: 1.6 });
[[296, 'base'], [356, 'SFT'], [416, 'DPO'], [476, 'RLVR']].forEach(([x, s]) => { POLY([[x, RAILY - 5], [x, RAILY + 5]], { sw: 1.6 }); T(s, x, 468, { align: 'C' }); });

FR.exportSettings = [{ format: 'PDF', contentsOnly: true, suffix: '', colorProfile: 'DOCUMENT' }];
out.frame = FR.id; out.nodes = FR.findAll(() => true).length; out.texts = FR.findAll(n => n.type === 'TEXT').length;
return out;
