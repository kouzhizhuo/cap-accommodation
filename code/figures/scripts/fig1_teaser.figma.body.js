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
