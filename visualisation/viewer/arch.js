// Onglet Architecture : schema zoomable, colore par les mesures layer/ lues dans MLflow.
//
// Le schema (visualisation/arch/<nom>.json) est un arbre de blocs. Chaque bloc a une
// taille fixe calculee une fois. Le zoom est semantique : un bloc conteneur s'ouvre
// quand ses enfants deviennent assez grands a l'ecran, sinon il reste une carte
// fermee dont le titre grossit avec la boite, donc lisible meme de loin.

import { $, api, h, s, icon, fmt, rampColor, divergingColor, RAMP, FAMILY, INHERIT, KIND_LABEL, store } from './common.js';

const OPS = new Set(['add', 'mul', 'concat', 'split']);
const G = {
  leafH: 50, line: 15, minW: 150, maxW: 380, op: 34,
  pad: 22, stub: 30, stubRow: 40, gapCol: 44, gapRow: 74, gapGrid: 14, lane: 42,
};
// un conteneur s'ouvre quand son plus grand enfant depasse cette hauteur a l'ecran
const OPEN_CHILD_PX = 38;
const OPEN_MIN_W_PX = 170;
const STATS = {
  grad_norm: 'Norme des gradients', update_ratio: 'Ratio de mise à jour', act_rms: 'RMS des activations',
  weight_norm: 'Norme des poids', tokens: 'Charge (tokens reçus)',
};

const st = {
  arch: null, root: null, byId: new Map(), list: [],
  view: { k: 1, x: 0, y: 0 }, kMin: 0.01,
  selected: null, panelOpen: false,
  runs: [], run: null, metric: 'grad_norm', scale: 'role', si: 0,
  playTimer: null, liveTimer: null,
};
let svg, vp, nodesLayer, linksLayer, stage;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const isOp = n => OPS.has(n.kind) && !n.children.length;
const isGroup = n => n.children.length > 0 && n.box === false;
const isBox = n => n.children.length > 0 && n.box !== false;

// --- mesure du texte ---------------------------------------------------------

const ctx = document.createElement('canvas').getContext('2d');
function textW(text, size, weight = 400) {
  ctx.font = `${weight} ${size}px Inter, "Segoe UI", system-ui, sans-serif`;
  return ctx.measureText(String(text)).width;
}

// --- preparation et mise en page ----------------------------------------------

function prepare(root) {
  st.byId.clear();
  st.list = [];
  const visit = (n, parent, depth) => {
    n.parent = parent;
    n.depth = depth;
    n.children = n.children || [];
    n.info = n.info || [];
    const inherit = INHERIT.has(n.kind) || n.box === false;
    n.acc = inherit && parent ? parent.acc : (FAMILY[n.kind] || '#64748B');
    n.exp = undefined;
    st.byId.set(n.id, n);
    st.list.push(n);
    n.children.forEach(c => visit(c, n, depth + 1));
  };
  visit(root, null, 0);
  // parametres d'un bloc sans module : somme de ses enfants (des freres ne se recouvrent pas)
  const total = n => {
    const kids = n.children.map(total);
    n.pTotal = n.params ?? (n.children.length ? kids.reduce((a, b) => a + b, 0) : null);
    return n.pTotal ?? 0;
  };
  total(root);
}

function measure(n) {
  // scale grossit une feuille entiere (cadre et texte), pour les blocs lus de loin
  const sc = n.scale || 1;
  if (isOp(n)) { n.w = n.h = G.op * sc; return; }
  if (!n.children.length) {
    const badge = n.badge ? textW(n.badge, 10.5, 600) + 20 : 0;
    const tw = Math.max(textW(n.label, 13, 600) + badge, ...n.info.map(t => textW(t, 11)), 0);
    // 54 de plus : la place de la valeur mesuree, a droite de la derniere ligne
    n.w = clamp(tw + 58 + 54, G.minW, G.maxW) * sc;
    n.h = Math.max(G.leafH, 30 + n.info.length * G.line + 6) * sc;
    return;
  }
  n.children.forEach(measure);
  const kids = n.children;
  const L = n.layout || 'col';
  const lane = n.skip ? G.lane : 0;
  let iw, ih;
  if (L === 'row') {
    iw = kids.reduce((a, c) => a + c.w, 0) + G.gapRow * (kids.length - 1);
    ih = Math.max(...kids.map(c => c.h));
  } else if (L === 'grid') {
    const cols = n.cols || Math.ceil(Math.sqrt(kids.length));
    const cw = Math.max(...kids.map(c => c.w)), ch = Math.max(...kids.map(c => c.h));
    const rows = Math.ceil(kids.length / cols);
    n.cell = { cols, cw, ch };
    iw = cols * cw + (cols - 1) * G.gapGrid;
    ih = rows * ch + (rows - 1) * G.gapGrid;
  } else {
    iw = Math.max(...kids.map(c => c.w)) + lane;
    ih = kids.reduce((a, c) => a + c.h, 0) + G.gapCol * (kids.length - 1);
  }
  n.iw = iw;
  n.ih = ih;
  n.maxKidH = Math.max(...kids.map(c => (isGroup(c) ? c.maxKidH || c.h : c.h)));
  if (isGroup(n)) { n.w = iw; n.h = ih; n.head = 0; return; }
  // le titre d'un cadre grossit avec la boite : il reste lisible quand on dezoome
  n.hf = clamp(Math.sqrt(iw * ih) * 0.011, 14, 150);
  n.head = n.hf * 2.4 + (n.info.length ? n.hf * 1.05 : 0);
  const padX = L === 'row' ? G.stubRow : G.pad;
  const padY = L === 'col' ? G.stub : G.pad;
  const hw = textW(n.label, n.hf, 700) + n.hf * 2.4 + (n.badge ? textW(n.badge, n.hf * 0.7, 600) + n.hf * 1.6 : 0) + n.hf * 6;
  const infoW = n.info.length ? textW(n.info.join('   ·   '), n.hf * 0.74) + n.hf * 2.4 : 0;
  n.w = Math.max(iw + 2 * padX, hw + 2 * G.pad, infoW + 2 * G.pad);
  n.h = n.head + ih + 2 * padY;
}

function place(n, x, y) {
  n.x = x;
  n.y = y;
  if (!n.children.length) return;
  const L = n.layout || 'col';
  let ix = x, iy = y;
  if (!isGroup(n)) {
    const padY = L === 'col' ? G.stub : G.pad;
    ix = x + (n.w - n.iw) / 2;
    iy = y + n.head + padY;
  }
  n.ix = ix;
  n.iy = iy;
  const lane = n.skip ? G.lane : 0;
  if (L === 'row') {
    let cx = ix;
    for (const c of n.children) { place(c, cx, iy + (n.ih - c.h) / 2); cx += c.w + G.gapRow; }
  } else if (L === 'grid') {
    const { cols, cw, ch } = n.cell;
    n.children.forEach((c, i) => place(c,
      ix + (i % cols) * (cw + G.gapGrid) + (cw - c.w) / 2,
      iy + Math.floor(i / cols) * (ch + G.gapGrid) + (ch - c.h) / 2));
  } else {
    let cy = iy;
    for (const c of n.children) { place(c, ix + lane + (n.iw - lane - c.w) / 2, cy); cy += c.h + G.gapCol; }
  }
}

// --- fleches -----------------------------------------------------------------

function entries(n, dir) {
  if (isGroup(n)) return n.flow === 'par' ? n.children.flatMap(c => entries(c, dir)) : entries(n.children[0], dir);
  return [dir === 'row' ? { x: n.x, y: n.y + n.h / 2 } : { x: n.x + n.w / 2, y: n.y }];
}
function exits(n, dir) {
  if (isGroup(n)) return n.flow === 'par' ? n.children.flatMap(c => exits(c, dir)) : exits(n.children.at(-1), dir);
  return [dir === 'row' ? { x: n.x + n.w, y: n.y + n.h / 2 } : { x: n.x + n.w / 2, y: n.y + n.h }];
}
function outOf(n) {
  if (n.out) return n.out;
  if (isGroup(n) && n.flow !== 'par') return outOf(n.children.at(-1));
  return null;
}

function elbow(pts, r = 9) {
  let d = `M${pts[0].x},${pts[0].y}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const a = pts[i - 1], b = pts[i], c = pts[i + 1];
    const lab = Math.hypot(b.x - a.x, b.y - a.y), lbc = Math.hypot(c.x - b.x, c.y - b.y);
    if (lab < 0.01 || lbc < 0.01) continue;
    const rr = Math.min(r, lab / 2, lbc / 2);
    const p1 = { x: b.x + (a.x - b.x) * rr / lab, y: b.y + (a.y - b.y) * rr / lab };
    const p2 = { x: b.x + (c.x - b.x) * rr / lbc, y: b.y + (c.y - b.y) * rr / lbc };
    d += `L${p1.x},${p1.y}Q${b.x},${b.y} ${p2.x},${p2.y}`;
  }
  const z = pts.at(-1);
  return `${d}L${z.x},${z.y}`;
}

function route(p, q, dir, mid) {
  if (dir === 'row') {
    if (Math.abs(p.y - q.y) < 0.5) return `M${p.x},${p.y}H${q.x}`;
    return elbow([p, { x: mid, y: p.y }, { x: mid, y: q.y }, q]);
  }
  if (Math.abs(p.x - q.x) < 0.5) return `M${p.x},${p.y}V${q.y}`;
  return elbow([p, { x: p.x, y: mid }, { x: q.x, y: mid }, q]);
}

function arrowHead(q, dir) {
  const rot = dir === 'row' ? 0 : 90;
  return s('g', { transform: `translate(${q.x},${q.y}) rotate(${rot})` },
    s('path', { class: 'ah', d: 'M0,0L-8,-4L-8,4Z' }));
}

// fleches d'un groupe de sorties vers un groupe d'entrees, coudes alignes
function connect(layer, outs, ins, dir, label) {
  const mid = dir === 'row'
    ? (Math.max(...outs.map(p => p.x)) + Math.min(...ins.map(p => p.x))) / 2
    : (Math.max(...outs.map(p => p.y)) + Math.min(...ins.map(p => p.y))) / 2;
  const heads = new Set();
  for (const p of outs) for (const q of ins) {
    layer.append(s('path', { class: 'edge', d: route(p, q, dir, mid) }));
    const key = `${q.x},${q.y}`;
    if (!heads.has(key)) { heads.add(key); layer.append(arrowHead(q, dir)); }
  }
  if (label && outs.length === 1) {
    const p = outs[0];
    const pos = dir === 'row'
      ? { x: p.x + 6, y: p.y - 7 }
      : { x: p.x + 7, y: Math.min(p.y + 15, (p.y + Math.min(...ins.map(q => q.y))) / 2 + 4) };
    layer.append(s('text', { class: 'elbl', x: pos.x, y: pos.y, 'font-size': 10.5 }, label));
  }
}

function drawEdges(n, layer) {
  const kids = n.children;
  const L = n.layout || 'col';
  const dir = L === 'row' ? 'row' : 'col';
  if (n.flow !== 'par' && n.flow !== 'none' && L !== 'grid') {
    for (let i = 0; i < kids.length - 1; i++) {
      connect(layer, exits(kids[i], dir), entries(kids[i + 1], dir), dir, outOf(kids[i]));
    }
  }
  // fleches d'entree et de sortie a l'interieur d'un cadre
  if (isBox(n) && n.flow !== 'none' && L !== 'grid') {
    const firsts = n.flow === 'par' ? kids.flatMap(c => entries(c, dir)) : entries(kids[0], dir);
    const lasts = n.flow === 'par' ? kids.flatMap(c => exits(c, dir)) : exits(kids.at(-1), dir);
    const S = dir === 'row' ? { x: n.x, y: n.iy + n.ih / 2 } : { x: n.x + n.w / 2, y: n.y + n.head };
    const E = dir === 'row' ? { x: n.x + n.w, y: n.iy + n.ih / 2 } : { x: n.x + n.w / 2, y: n.y + n.h };
    connect(layer, [S], firsts, dir, null);
    const mid = dir === 'row' ? (Math.max(...lasts.map(p => p.x)) + E.x) / 2 : (Math.max(...lasts.map(p => p.y)) + E.y) / 2;
    for (const p of lasts) layer.append(s('path', { class: 'edge', d: route(p, E, dir, mid) }));
  }
  // connexion residuelle : du debut du groupe jusqu'a l'addition
  if (n.skip && kids.length > 1) {
    const tgt = n.skip === true ? kids.at(-1) : kids.find(c => c.id === n.skip || c.id.endsWith(`/${n.skip}`)) || kids.at(-1);
    const e = entries(kids[0], 'col')[0];
    const by = e.y - Math.min(G.gapCol, G.stub) * 0.55;
    const laneX = (isGroup(n) ? n.x : n.ix) + G.lane * 0.38;
    const ty = tgt.y + tgt.h / 2;
    layer.append(s('path', { class: 'skip', d: elbow([{ x: e.x, y: by }, { x: laneX, y: by }, { x: laneX, y: ty }, { x: tgt.x, y: ty }]) }));
    layer.append(s('g', { transform: `translate(${tgt.x},${ty})` }, s('path', { class: 'ah', d: 'M0,0L-8,-4L-8,4Z' })));
    layer.append(s('g', { transform: `translate(${e.x},${by})` }, s('circle', { class: 'skipdot', r: 3.2 })));
  }
}

// --- rendu des blocs ------------------------------------------------------------

function renderNode(n) {
  const g = s('g', { class: 'n', 'data-id': n.id, style: `--acc:${n.acc}` });
  n.g = g;
  if (isOp(n)) {
    g.append(opBody(n));
  } else if (!n.children.length) {
    g.append(leafBody(n));
  } else if (isGroup(n)) {
    g.classList.add('exp');
    g.append(kidsLayer(n));
  } else {
    g.append(closedBody(n), frame(n), kidsLayer(n));
  }
  return g;
}

function kidsLayer(n) {
  const g = s('g', { class: 'kids' });
  const edges = s('g', { class: 'edges' });
  drawEdges(n, edges);
  g.append(edges, ...n.children.map(renderNode));
  return g;
}

function badge(text, x, y, size) {
  const w = textW(text, size, 600) + size * 1.2;
  return s('g', { class: 'badge', transform: `translate(${x - w},${y})` },
    s('rect', { width: w, height: size * 1.7, rx: size * 0.85 }),
    s('text', { x: w / 2, y: size * 1.22, 'font-size': size, 'text-anchor': 'middle' }, text));
}

function leafBody(n) {
  const sc = n.scale || 1;
  const w = n.w / sc, hh = n.h / sc;
  const g = s('g', { class: 'body', transform: `translate(${n.x},${n.y}) scale(${sc})` });
  const ty = n.info.length ? 21 : hh / 2 + 4.5;
  g.append(
    s('rect', { class: 'card', width: w, height: hh, rx: 10, filter: 'url(#shadow)' }),
    s('rect', { class: 'bar', x: 0, y: 7, width: 4, height: hh - 14, rx: 2 }),
    icon(n.kind, { class: 'ico', x: 13, y: n.info.length ? 12 : (hh - 20) / 2, width: 20, height: 20 }),
    s('text', { class: 'lbl', x: 42, y: ty, 'font-size': 13 }, n.label),
    ...n.info.map((t, i) => s('text', { class: 'inf', x: 42, y: ty + G.line * (i + 1), 'font-size': 11 }, t)),
    s('text', { class: 'mval', x: w - 10, y: hh - 8, 'font-size': 10.5, 'text-anchor': 'end' }),
  );
  if (n.badge) g.append(badge(n.badge, w - 8, 8, 10));
  return g;
}

function opBody(n) {
  const r = n.w / 2, cx = n.x + r, cy = n.y + r, a = r * 0.45;
  const g = s('g', { class: 'body opb' });
  g.append(s('circle', { class: 'op card', cx, cy, r }));
  if (n.kind === 'add') g.append(s('path', { class: 'opsym', d: `M${cx - a},${cy}H${cx + a}M${cx},${cy - a}V${cy + a}` }));
  else if (n.kind === 'mul') g.append(s('circle', { class: 'opsym', cx, cy, r: a * 0.9 }), s('circle', { class: 'opdot', cx, cy, r: 2 }));
  else if (n.kind === 'concat') g.append(s('path', { class: 'opsym', d: `M${cx - a * 0.4},${cy - a}V${cy + a}M${cx + a * 0.4},${cy - a}V${cy + a}` }));
  else g.append(s('path', { class: 'opsym', d: `M${cx},${cy + a}V${cy}L${cx - a},${cy - a}M${cx},${cy}L${cx + a},${cy - a}` }));
  g.append(s('text', { class: 'mval', x: cx + r + 4, y: cy + 4, 'font-size': 10.5 }));
  return g;
}

// carte d'un conteneur ferme : titre dimensionne sur la boite
function closedBody(n) {
  const g = s('g', { class: 'body', transform: `translate(${n.x},${n.y})` });
  const lines = n.info.slice(0, 3);
  const F = Math.max(13, Math.min(n.h * 0.11, (n.w * 0.8) / Math.max(5, n.label.length * 0.6)));
  const fi = F * 0.5;
  const sub = `${fmt.params(n.pTotal)} paramètres · ${countInside(n)} blocs`;
  const blockH = F * 1.9 + F * 1.25 + lines.length * fi * 1.4 + fi * 1.6;
  let y = (n.h - blockH) / 2;
  const rx = Math.min(16, F * 0.7);
  g.append(
    s('rect', { class: 'card', width: n.w, height: n.h, rx, filter: 'url(#shadow)' }),
    s('rect', { class: 'bar', x: 0, y: rx * 0.6, width: Math.max(4, F * 0.18), height: n.h - rx * 1.2, rx: 2 }),
    icon(n.kind, { class: 'ico', x: n.w / 2 - F * 0.85, y, width: F * 1.7, height: F * 1.7 }),
  );
  y += F * 1.9 + F;
  g.append(s('text', { class: 'lbl', x: n.w / 2, y, 'font-size': F, 'text-anchor': 'middle' }, n.label));
  for (const t of lines) {
    y += fi * 1.4;
    g.append(s('text', { class: 'inf', x: n.w / 2, y, 'font-size': fi, 'text-anchor': 'middle' }, t));
  }
  y += fi * 1.6;
  g.append(s('text', { class: 'sub', x: n.w / 2, y, 'font-size': fi * 0.9, 'text-anchor': 'middle' }, sub));
  g.append(s('text', { class: 'mval', x: n.w / 2, y: y + fi * 1.7, 'font-size': fi * 1.05, 'text-anchor': 'middle' }));
  if (n.badge) g.append(badge(n.badge, n.w - F * 0.5, F * 0.5, F * 0.6));
  return g;
}

function frame(n) {
  const f = n.hf;
  const g = s('g', { class: 'frame' });
  g.append(s('rect', { class: 'fr', x: n.x, y: n.y, width: n.w, height: n.h, rx: Math.min(18, f * 0.9) }));
  const hdr = s('g', { class: 'hdr' });
  const tx = n.x + G.pad + f * 1.9;
  const ty = n.y + f * 1.7;
  hdr.append(
    s('rect', { x: n.x, y: n.y, width: n.w, height: n.head, fill: 'transparent' }),
    icon(n.kind, { class: 'ico', x: n.x + G.pad, y: n.y + f * 0.55, width: f * 1.45, height: f * 1.45 }),
    s('text', { class: 'lbl', x: tx, y: ty, 'font-size': f, 'font-weight': 700 }, n.label),
  );
  if (n.info.length) {
    hdr.append(s('text', { class: 'inf', x: tx, y: ty + f * 1.05, 'font-size': f * 0.74 }, n.info.join('   ·   ')));
  }
  const right = n.pTotal ? `${fmt.params(n.pTotal)} params` : '';
  if (right) hdr.append(s('text', { class: 'sub', x: n.x + n.w - G.pad, y: ty, 'font-size': f * 0.72, 'text-anchor': 'end' }, right));
  if (n.badge) hdr.append(badge(n.badge, n.x + n.w - G.pad, ty + f * 0.45, f * 0.62));
  g.append(hdr);
  return g;
}

function countInside(n) {
  let c = 0;
  const walk = m => m.children.forEach(k => { if (!isGroup(k)) c++; walk(k); });
  walk(n);
  return c;
}

// --- zoom semantique ------------------------------------------------------------

let lodQueued = false;
function requestLOD() {
  if (lodQueued) return;
  lodQueued = true;
  requestAnimationFrame(() => { lodQueued = false; updateLOD(); });
}

function visibleWorld() {
  const { k, x, y } = st.view;
  return { x0: -x / k, y0: -y / k, x1: (stage.clientWidth - x) / k, y1: (stage.clientHeight - y) / k };
}
const intersects = (n, v) => n.x < v.x1 && n.x + n.w > v.x0 && n.y < v.y1 && n.y + n.h > v.y0;

function updateLOD() {
  if (!st.root) return;
  const { k } = st.view;
  svg.style.setProperty('--ik', String(1 / k));
  const vis = visibleWorld();
  const walk = n => {
    if (!n.children.length) return;
    let open = true;
    if (!isGroup(n) && n !== st.root) {
      open = intersects(n, vis) && n.w * k >= OPEN_MIN_W_PX && n.maxKidH * k >= OPEN_CHILD_PX;
    }
    if (open !== n.exp) { n.exp = open; n.g.classList.toggle('exp', open); }
    if (open) n.children.forEach(walk);
  };
  walk(st.root);
  drawLinks();
  updateCrumbs();
  drawMinimapView();
  if (st.run && st.scale === 'global') recolor();
}

// bloc qui represente n a l'ecran : lui-meme, ou son ancetre ferme le plus haut
function shown(n) {
  let rep = n;
  for (let a = n.parent; a; a = a.parent) if (!isGroup(a) && a.exp === false) rep = a;
  return rep;
}
function isShown(n) {
  for (let a = n.parent; a; a = a.parent) if (!isGroup(a) && a.exp === false) return false;
  return true;
}

// --- liens hors flux (poids partages, cross-attention, MTP) ----------------------

function drawLinks() {
  linksLayer.replaceChildren();
  const done = new Set();
  const vis = visibleWorld();
  for (const lk of st.arch.links || []) {
    const a = st.byId.get(lk.from), b = st.byId.get(lk.to);
    if (!a || !b) continue;
    const A = shown(a), B = shown(b);
    if (A === B || isGroup(A) || isGroup(B)) continue;
    // aucune extremite a l'ecran : le lien ne ferait que traverser la vue
    if (!intersects(A, vis) && !intersects(B, vis)) continue;
    const key = `${A.id}>${B.id}:${lk.label}`;
    if (done.has(key)) continue;
    done.add(key);
    const color = FAMILY[lk.kind] || '#64748B';
    const ac = { x: A.x + A.w / 2, y: A.y + A.h / 2 }, bc = { x: B.x + B.w / 2, y: B.y + B.h / 2 };
    let p, q, c1, c2, rot;
    const dx = bc.x - ac.x, dy = bc.y - ac.y;
    const sel = st.selected && (A === shown(st.selected) || B === shown(st.selected));
    if (Math.abs(dy) < Math.abs(dx) * 0.3 && Math.abs(dx) > 3 * Math.max(A.w, B.w)) {
      // deux blocs eloignes sur la meme ligne : arc par-dessus, sans traverser ce qu'il y a entre eux
      const top = Math.min(A.y, B.y);
      const lift = clamp(Math.abs(dx) * 0.08, 40, 900);
      p = { x: ac.x, y: A.y };
      q = { x: bc.x, y: B.y };
      c1 = { x: p.x, y: top - lift }; c2 = { x: q.x, y: top - lift }; rot = 90;
    } else if (Math.abs(dy) > Math.abs(dx) * 0.6) {
      const down = dy > 0;
      p = { x: ac.x, y: down ? A.y + A.h : A.y };
      q = { x: bc.x, y: down ? B.y : B.y + B.h };
      const m = (q.y - p.y) * 0.5;
      c1 = { x: p.x, y: p.y + m }; c2 = { x: q.x, y: q.y - m }; rot = down ? 90 : -90;
    } else {
      const right = dx > 0;
      p = { x: right ? A.x + A.w : A.x, y: ac.y };
      q = { x: right ? B.x : B.x + B.w, y: bc.y };
      const m = (q.x - p.x) * 0.5;
      c1 = { x: p.x + m, y: p.y }; c2 = { x: q.x - m, y: q.y }; rot = right ? 0 : 180;
    }
    linksLayer.append(s('path', { class: `lk${sel ? ' on' : ''}`, d: `M${p.x},${p.y}C${c1.x},${c1.y} ${c2.x},${c2.y} ${q.x},${q.y}`, stroke: color }));
    linksLayer.append(s('g', { transform: `translate(${q.x},${q.y}) rotate(${rot})` },
      s('path', { class: 'lk-ah', d: 'M0,0L-9,-4.5L-9,4.5Z', fill: color })));
    if (lk.label) {
      // libelle a taille constante a l'ecran, au milieu de la courbe
      const mx = 0.125 * p.x + 0.375 * c1.x + 0.375 * c2.x + 0.125 * q.x;
      const my = 0.125 * p.y + 0.375 * c1.y + 0.375 * c2.y + 0.125 * q.y;
      const fs = 11 / st.view.k, w = textW(lk.label, 11, 600) / st.view.k + 14 / st.view.k;
      linksLayer.append(s('g', { class: 'lk-lbl', transform: `translate(${mx - w / 2},${my - fs})` },
        s('rect', { width: w, height: fs * 1.9, rx: fs * 0.9, stroke: color }),
        s('text', { x: w / 2, y: fs * 1.3, 'font-size': fs, 'text-anchor': 'middle', fill: color }, lk.label)));
    }
  }
}

// --- vue : zoom, deplacement, animation ------------------------------------------

function setView(k, x, y) {
  st.view = { k, x, y };
  vp.setAttribute('transform', `translate(${x},${y}) scale(${k})`);
  requestLOD();
}

function zoomAt(px, py, factor) {
  const { k, x, y } = st.view;
  const k2 = clamp(k * factor, st.kMin, 8);
  setView(k2, px - (px - x) * (k2 / k), py - (py - y) * (k2 / k));
}

function viewFor(n, margin = 0.9) {
  const right = st.panelOpen ? 384 : 0;
  const vw = stage.clientWidth - right, vh = stage.clientHeight;
  const k = clamp(Math.min((vw * margin) / n.w, (vh * margin) / n.h), st.kMin, 8);
  return { k, x: vw / 2 - (n.x + n.w / 2) * k, y: vh / 2 - (n.y + n.h / 2) * k };
}

let anim = null;
function zoomTo(n, ms = 480) {
  // le resume du modele masquerait le haut du bloc vise
  $('#arch-summary').classList.toggle('folded', n !== st.root);
  const to = viewFor(n);
  const from = { ...st.view };
  cancelAnimationFrame(anim);
  const t0 = performance.now();
  // interpolation sur le centre du monde et sur log(k), pour un zoom regulier
  const cw = stage.clientWidth / 2, ch = stage.clientHeight / 2;
  const c0 = { x: (cw - from.x) / from.k, y: (ch - from.y) / from.k };
  const c1 = { x: (cw - to.x) / to.k, y: (ch - to.y) / to.k };
  const step = now => {
    const t = Math.min(1, (now - t0) / ms);
    const e = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    const k = Math.exp(Math.log(from.k) + (Math.log(to.k) - Math.log(from.k)) * e);
    const cx = c0.x + (c1.x - c0.x) * e, cy = c0.y + (c1.y - c0.y) * e;
    setView(k, cw - cx * k, ch - cy * k);
    if (t < 1) anim = requestAnimationFrame(step);
  };
  anim = requestAnimationFrame(step);
}

function fitAll(ms = 0) {
  const v = viewFor(st.root, 0.94);
  st.kMin = v.k * 0.5;
  if (ms) zoomTo(st.root, ms); else setView(v.k, v.x, v.y);
}

// conteneur ouvert le plus profond sous le centre de l'ecran
function focusNode() {
  const { k, x, y } = st.view;
  const cx = (stage.clientWidth / 2 - x) / k, cy = (stage.clientHeight / 2 - y) / k;
  let cur = st.root;
  for (;;) {
    const next = cur.children.find(c => c.children.length && (isGroup(c) || c.exp)
      && cx >= c.x && cx <= c.x + c.w && cy >= c.y && cy <= c.y + c.h);
    if (!next) return cur;
    cur = next;
  }
}

function updateCrumbs() {
  const el = $('#arch-crumbs');
  const chain = [];
  for (let n = st.selected && isShown(st.selected) ? st.selected : focusNode(); n; n = n.parent) {
    if (!isGroup(n)) chain.unshift(n);
  }
  el.replaceChildren(...chain.flatMap((n, i) => [
    i ? h('span', {}, '›') : null,
    h('button', { onclick: () => zoomTo(n.children.length ? n : n.parent || n) }, n.label),
  ]).filter(Boolean));
}

// --- minimap ------------------------------------------------------------------------

function drawMinimap() {
  const mm = $('#minimap svg');
  const r = st.root;
  mm.setAttribute('viewBox', `${r.x - 40} ${r.y - 40} ${r.w + 80} ${r.h + 80}`);
  mm.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const items = [];
  const walk = (n, d) => {
    for (const c of n.children) {
      if (isGroup(c)) { walk(c, d); continue; }
      if (c.kind === 'add' || c.kind === 'mul') continue;
      items.push(s('rect', { class: 'mm', x: c.x, y: c.y, width: c.w, height: c.h, rx: 6, style: `--acc:${c.acc}` }));
      if (d < 2 && c.children.length) walk(c, d + 1);
    }
  };
  walk(r, 0);
  mm.replaceChildren(...items, s('rect', { class: 'mv', id: 'mm-view' }));
}
function drawMinimapView() {
  const v = visibleWorld();
  const rect = $('#mm-view');
  if (!rect) return;
  rect.setAttribute('x', v.x0); rect.setAttribute('y', v.y0);
  rect.setAttribute('width', v.x1 - v.x0); rect.setAttribute('height', v.y1 - v.y0);
}
function minimapJump(ev) {
  const mm = $('#minimap svg');
  const pt = mm.createSVGPoint();
  pt.x = ev.clientX; pt.y = ev.clientY;
  const w = pt.matrixTransform(mm.getScreenCTM().inverse());
  const { k } = st.view;
  setView(k, stage.clientWidth / 2 - w.x * k, stage.clientHeight / 2 - w.y * k);
}

// --- mesures de l'entrainement ----------------------------------------------------

function unitsOf(n) {
  if (n._units) return n._units;
  const run = st.run;
  let units = [];
  if (n.module) {
    const m = n.module;
    units = run.paramUnits.filter(u => u === m || u.startsWith(`${m}.`));
  }
  if (!units.length && !n.module) units = [...new Set(n.children.flatMap(unitsOf))];
  n._units = units;
  return units;
}

function nodeValue(n, stat, si) {
  const V = st.run.values;
  if (n.module && V[`${n.module}/${stat}`]) {
    const v = V[`${n.module}/${stat}`][si];
    if (stat === 'tokens') return expertRatio(n.module, si);
    return v;
  }
  if (stat === 'act_rms' || stat === 'tokens') return null;
  const units = unitsOf(n);
  if (!units.length) return null;
  if (stat === 'update_ratio') {
    let num = 0, den = 0, any = false;
    for (const u of units) {
      const r = V[`${u}/update_ratio`]?.[si], w = V[`${u}/weight_norm`]?.[si];
      if (r == null || w == null) continue;
      num += (r * w) ** 2; den += w * w; any = true;
    }
    return any && den > 0 ? Math.sqrt(num / den) : null;
  }
  let sum = 0, any = false;
  for (const u of units) {
    const v = V[`${u}/${stat}`]?.[si];
    if (v == null) continue;
    sum += v * v; any = true;
  }
  return any ? Math.sqrt(sum) : null;
}

// charge d'un expert rapportee a la moyenne de sa couche (1 = parfaitement equilibre)
function expertRatio(module, si) {
  const parent = module.slice(0, module.lastIndexOf('.'));
  const run = st.run;
  const sibs = run.expertGroups.get(parent) || [];
  let total = 0;
  for (const u of sibs) total += run.values[`${u}/tokens`]?.[si] || 0;
  const mean = total / Math.max(1, sibs.length);
  const v = run.values[`${module}/tokens`]?.[si];
  return v == null || !mean ? null : v / mean;
}

const roleOf = n => (n.module || n.id).replace(/\d+/g, '#');

function displayedNodes() {
  const out = [];
  const walk = n => {
    for (const c of n.children) {
      if (isGroup(c) || (c.children.length && c.exp)) walk(c);
      else out.push(c);
    }
  };
  walk(st.root);
  return out;
}

function recolor() {
  const run = st.run;
  const stat = st.metric;
  for (const n of st.list) {
    if (isGroup(n) || !n.g) continue;
    const body = n.g.querySelector(':scope > .body');
    if (!body) continue;
    const card = body.querySelector('.card');
    const mval = body.querySelector('.mval');
    n._v = run ? nodeValue(n, stat, st.si) : null;
    if (!run) {
      card.style.fill = ''; n.g.classList.remove('hot', 'dim'); if (mval) mval.textContent = '';
    }
  }
  renderLegend();
  if (!run) return;

  const diverging = stat === 'tokens';
  const domains = new Map();
  const pool = st.scale === 'global' ? displayedNodes() : st.list;
  for (const n of pool) {
    if (isGroup(n) || n._v == null || !(n._v > 0)) continue;
    const key = st.scale === 'global' ? 'all' : roleOf(n);
    const d = domains.get(key) || [Infinity, -Infinity];
    d[0] = Math.min(d[0], n._v); d[1] = Math.max(d[1], n._v);
    domains.set(key, d);
  }
  st.domain = domains.get('all');
  for (const n of st.list) {
    if (isGroup(n) || !n.g) continue;
    const body = n.g.querySelector(':scope > .body');
    if (!body) continue;
    const card = body.querySelector('.card');
    const mval = body.querySelector('.mval');
    const v = n._v;
    if (v == null || !(v > 0) && !diverging) {
      card.style.fill = '';
      n.g.classList.toggle('dim', !isOp(n));
      n.g.classList.remove('hot');
      if (mval) mval.textContent = '';
      continue;
    }
    let col;
    if (diverging) {
      col = divergingColor(clamp(Math.log2(Math.max(v, 1e-3)) / 2, -1, 1));
    } else {
      const d = domains.get(st.scale === 'global' ? 'all' : roleOf(n));
      const lo = Math.log(d[0]), hi = Math.log(d[1]);
      col = rampColor(hi - lo < 1e-9 ? 0.5 : (Math.log(v) - lo) / (hi - lo));
    }
    card.style.fill = col.color;
    n.g.classList.remove('dim');
    n.g.classList.toggle('hot', col.dark && !isOp(n));
    if (mval) mval.textContent = diverging ? `${v.toFixed(2)}×` : fmt.num(v, 3);
  }
}

function renderLegend() {
  const el = $('#arch-legend');
  if (!st.run) { el.replaceChildren(); return; }
  const stat = st.metric;
  if (stat === 'tokens') {
    el.replaceChildren(h('span', {}, '¼×'),
      h('span', { class: 'bar', style: { background: `linear-gradient(90deg, #2a78d6, var(--div-mid), #e34948)` } }),
      h('span', {}, '4×'), h('span', {}, 'charge / moyenne de la couche'));
    return;
  }
  const grad = `linear-gradient(90deg, ${RAMP.join(',')})`;
  if (st.scale === 'global' && st.domain) {
    el.replaceChildren(h('span', { class: 'mono' }, fmt.num(st.domain[0], 3)),
      h('span', { class: 'bar', style: { background: grad } }),
      h('span', { class: 'mono' }, fmt.num(st.domain[1], 3)), h('span', {}, `${STATS[stat]}, échelle log`));
  } else {
    el.replaceChildren(h('span', {}, 'plus faible'), h('span', { class: 'bar', style: { background: grad } }),
      h('span', {}, 'plus fort'), h('span', {}, 'que le même bloc des autres couches'));
  }
}

async function loadRun(runId) {
  stopPlay();
  clearInterval(st.liveTimer);
  for (const n of st.list) n._units = null;
  if (!runId) { st.run = null; $('#arch-steps').hidden = true; recolor(); refreshPanel(); return; }
  const meta = st.runs.find(r => r.id === runId);
  const data = await api(`run/${runId}/layers`);
  st.run = { id: runId, name: meta?.name || runId, status: meta?.status, steps: data.steps, values: data.values };
  indexRun(st.run);
  st.si = Math.max(0, st.run.steps.length - 1);
  const slider = $('#arch-step');
  slider.max = String(Math.max(0, st.run.steps.length - 1));
  slider.value = String(st.si);
  $('#arch-steps').hidden = false;
  updateStepLabel();
  recolor();
  refreshPanel();
  if (meta?.status === 'RUNNING') st.liveTimer = setInterval(pollRun, 6000);
}

function indexRun(run) {
  const units = new Set(), experts = new Map();
  for (const key of Object.keys(run.values)) {
    const i = key.lastIndexOf('/');
    const unit = key.slice(0, i), stat = key.slice(i + 1);
    if (stat === 'weight_norm') units.add(unit);
    if (stat === 'tokens') {
      const parent = unit.slice(0, unit.lastIndexOf('.'));
      if (!experts.has(parent)) experts.set(parent, []);
      experts.get(parent).push(unit);
    }
  }
  run.paramUnits = [...units];
  run.expertGroups = experts;
}

async function pollRun() {
  const run = st.run;
  if (!run) return;
  const last = run.steps.at(-1) ?? -1;
  const data = await api(`run/${run.id}/layers?since=${last}`).catch(() => null);
  if (!data || !data.steps.length) return;
  const follow = st.si === run.steps.length - 1;
  const n0 = run.steps.length;
  run.steps.push(...data.steps);
  for (const [k, arr] of Object.entries(run.values)) {
    const add = data.values[k] || data.steps.map(() => null);
    arr.push(...add);
  }
  for (const [k, arr] of Object.entries(data.values)) {
    if (!run.values[k]) run.values[k] = [...Array(n0).fill(null), ...arr];
  }
  indexRun(run);
  $('#arch-step').max = String(run.steps.length - 1);
  if (follow) { st.si = run.steps.length - 1; $('#arch-step').value = String(st.si); }
  updateStepLabel();
  recolor();
  refreshPanel();
}

function updateStepLabel() {
  const run = st.run;
  if (!run || !run.steps.length) { $('#arch-step-label').textContent = 'aucune mesure layer/'; return; }
  $('#arch-step-label').textContent = `step ${fmt.int(run.steps[st.si])}  ·  ${st.si + 1}/${run.steps.length}`;
}

function stopPlay() {
  clearInterval(st.playTimer);
  st.playTimer = null;
  $('#arch-play').textContent = '▶';
}
function togglePlay() {
  if (st.playTimer) return stopPlay();
  if (!st.run) return;
  if (st.si >= st.run.steps.length - 1) st.si = 0;
  $('#arch-play').textContent = '❚❚';
  st.playTimer = setInterval(() => {
    if (st.si >= st.run.steps.length - 1) return stopPlay();
    st.si++;
    $('#arch-step').value = String(st.si);
    updateStepLabel(); recolor(); refreshPanel();
  }, 320);
}

// --- panneau de detail ------------------------------------------------------------

function select(n) {
  st.selected?.g?.classList.remove('sel');
  st.selected = n;
  n?.g?.classList.add('sel');
  st.panelOpen = !!n;
  refreshPanel();
  updateCrumbs();
  if (st.root) drawLinks();
}

function refreshPanel() {
  const panel = $('#arch-panel');
  const n = st.selected;
  if (!n) { panel.hidden = true; return; }
  panel.hidden = false;
  const total = st.arch.params_total;
  const rows = [];
  const row = (k, v) => v != null && v !== '' && rows.push(h('dt', {}, k), h('dd', {}, v));
  row('Module', n.module ? h('code', {}, n.module) : null);
  row('Classe', n.cls ? h('code', {}, n.cls) : null);
  if (n.pTotal != null) row('Paramètres', `${fmt.params(n.pTotal)}${total ? `  (${(100 * n.pTotal / total).toFixed(1)} %)` : ''}`);
  row('Entrée', n.in ? h('code', {}, n.in) : null);
  row('Sortie', n.out ? h('code', {}, n.out) : null);
  if (n.calls > 1) row('Appels', `${n.calls} par forward`);
  for (const line of n.info) row('·', line);

  const kids = [];
  kids.push(h('button', { class: 'close', title: 'Fermer', onclick: () => { select(null); } }, '×'));
  const ico = s('svg', { viewBox: '0 0 24 24', style: `color:${n.acc}` }, icon(n.kind, { class: 'ico', width: 24, height: 24 }));
  ico.querySelector('use').setAttribute('style', `stroke:${n.acc};fill:none;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round`);
  kids.push(h('h3', {}, ico, n.label));
  kids.push(h('span', { class: 'kind', style: { background: `color-mix(in srgb, ${n.acc} 14%, transparent)`, color: n.acc } }, KIND_LABEL[n.kind] || n.kind));
  if (n.badge) kids.push(h('span', { class: 'kind', style: { marginLeft: '6px', background: 'var(--hover)' } }, n.badge));
  kids.push(h('dl', {}, rows));
  if (n.doc) kids.push(h('p', { class: 'doc' }, n.doc));
  if (n.ref) kids.push(h('p', { class: 'muted small' }, `Référence : ${n.ref}`));
  if (n.children.length) {
    kids.push(h('button', { class: 'ghost', onclick: () => zoomTo(n) }, `Zoomer dessus (${countInside(n)} blocs)`));
  }
  if (st.run) kids.push(...metricsSection(n));
  panel.replaceChildren(...kids);
}

function metricsSection(n) {
  const out = [h('h4', {}, `Mesures · ${st.run.name}`)];
  const stats = ['grad_norm', 'update_ratio', 'act_rms', 'weight_norm', 'tokens'];
  let any = false;
  for (const stat of stats) {
    const series = st.run.steps.map((_, i) => nodeValue(n, stat, i));
    if (!series.some(v => v != null)) continue;
    any = true;
    out.push(sparkline(STATS[stat], series, stat === 'tokens'));
  }
  const loads = expertLoads(n);
  if (loads) out.push(loads);
  if (!any && !loads) out.push(h('p', { class: 'muted small' }, 'Pas de mesure layer/ rattachée à ce bloc dans ce run.'));
  return out;
}

function sparkline(name, series, ratio) {
  const W = 300, H = 44;
  const pts = series.map((v, i) => [i, v]).filter(p => p[1] != null);
  const vals = pts.map(p => p[1]);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const X = i => (series.length > 1 ? (i / (series.length - 1)) * W : W / 2);
  const Y = v => H - 4 - (hi - lo < 1e-12 ? 0.5 : (v - lo) / (hi - lo)) * (H - 8);
  const d = pts.map((p, j) => `${j ? 'L' : 'M'}${X(p[0])},${Y(p[1])}`).join('');
  const area = `${d}L${X(pts.at(-1)[0])},${H}L${X(pts[0][0])},${H}Z`;
  const cur = series[st.si];
  const svgEl = s('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none' },
    s('path', { class: 'a', d: area }), s('path', { class: 'l', d }));
  if (cur != null) svgEl.append(s('circle', { cx: X(st.si), cy: Y(cur), r: 3.5 }));
  const fmtv = v => (v == null ? '–' : ratio ? `${v.toFixed(2)}×` : fmt.num(v, 3));
  return h('div', { class: 'spark' }, h('span', { class: 'name' }, name),
    h('span', { class: 'val' }, fmtv(cur)), svgEl,
    h('span', { class: 'muted small' }, `min ${fmtv(lo)}`), h('span', { class: 'muted small' }, `max ${fmtv(hi)}`));
}

function expertLoads(n) {
  if (!n.module || !st.run.expertGroups.size) return null;
  const groups = [...st.run.expertGroups.entries()].filter(([p]) => p === n.module || p.startsWith(`${n.module}.`));
  if (!groups.length) return null;
  const [parent, units] = groups[0];
  const idx = u => Number(u.slice(u.lastIndexOf('.') + 1));
  const sorted = [...units].sort((a, b) => idx(a) - idx(b));
  const ratios = sorted.map(u => expertRatio(u, st.si) ?? 0);
  const W = 300, H = 110, cap = 4;
  const bw = W / ratios.length;
  const y1 = H - 10 - (1 / cap) * (H - 20);
  const bars = ratios.map((r, i) => {
    const hh = (Math.min(r, cap) / cap) * (H - 20);
    const c = divergingColor(clamp(Math.log2(Math.max(r, 1e-3)) / 2, -1, 1)).color;
    return s('rect', { x: i * bw + 0.5, y: H - 10 - hh, width: Math.max(1, bw - 1), height: hh, fill: c });
  });
  const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  const sd = Math.sqrt(ratios.reduce((a, b) => a + (b - mean) ** 2, 0) / ratios.length);
  return h('div', { class: 'loads' },
    h('h4', {}, `Charge des ${ratios.length} experts · step ${fmt.int(st.run.steps[st.si])}`),
    s('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none' }, ...bars,
      s('line', { x1: 0, x2: W, y1, y2: y1 })),
    h('p', { class: 'muted small' }, `${parent} · CV de charge ${(sd / (mean || 1)).toFixed(2)} (0 = équilibre parfait). Pointillé : charge moyenne.`));
}

// --- recherche ------------------------------------------------------------------------

function search(q) {
  const box = $('#arch-results');
  q = q.trim().toLowerCase();
  if (!q) { box.hidden = true; return; }
  const scored = [];
  for (const n of st.list) {
    if (isGroup(n)) continue;
    const label = n.label.toLowerCase(), mod = (n.module || '').toLowerCase();
    let sc = 0;
    if (label === q) sc = 4;
    else if (label.startsWith(q)) sc = 3;
    else if (label.includes(q)) sc = 2;
    else if (mod.includes(q) || n.info.join(' ').toLowerCase().includes(q)) sc = 1;
    if (sc) scored.push([sc, n]);
  }
  scored.sort((a, b) => b[0] - a[0] || a[1].depth - b[1].depth);
  const items = scored.slice(0, 40).map(([, n]) => h('button', {
    onclick: () => { box.hidden = true; goTo(n); },
  }, n.label, h('small', {}, n.module || pathLabel(n))));
  box.replaceChildren(...(items.length ? items : [h('p', { class: 'muted small', style: { padding: '6px 8px' } }, 'Aucun bloc')]));
  box.hidden = false;
}
function pathLabel(n) {
  const parts = [];
  for (let a = n.parent; a; a = a.parent) if (!isGroup(a)) parts.unshift(a.label);
  return parts.join(' › ');
}
function goTo(n) {
  select(n);
  const target = n.children.length ? n : (n.parent && isGroup(n.parent) ? n.parent : n.parent || n);
  zoomTo(target === st.root ? n : target);
}

// --- resume du modele ----------------------------------------------------------------

function renderSummary() {
  const a = st.arch;
  const el = $('#arch-summary');
  const cov = a.coverage || {};
  const pct = cov.total ? (100 * cov.covered) / cov.total : 100;
  const ok = !cov.missing?.length && !cov.unknown_modules?.length;
  const parts = [
    h('button', { class: 'toggle', title: 'Replier', onclick: () => el.classList.toggle('folded') }, '▾'),
    h('h2', {}, a.title),
    a.subtitle ? h('p', { class: 'subtitle' }, a.subtitle) : null,
    h('div', { class: 'chips' }, h('span', { class: 'chip' }, h('b', {}, fmt.params(a.params_total)), 'paramètres'),
      ...(a.summary || []).map(([k, v]) => h('span', { class: 'chip' }, k, h('b', {}, v)))),
    h('div', { class: `cov ${ok ? 'ok' : 'warn'}` }, ok
      ? `✓ Schéma complet : ${pct.toFixed(1)} % des paramètres rattachés à un bloc`
      : `⚠ ${Math.min(pct, 99.999).toFixed(3)} % des paramètres rattachés, ${cov.missing.length + cov.unknown_modules.length} module(s) hors schéma`),
  ];
  if (!ok) {
    parts.push(h('details', {}, h('summary', {}, 'Voir les modules oubliés'),
      ...cov.missing.map(m => h('div', {}, h('code', {}, m.module), ` · ${fmt.params(m.params)}`)),
      ...cov.unknown_modules.map(m => h('div', {}, h('code', {}, m), ' · module introuvable'))));
  }
  const never = cov.never_called || [];
  if (never.length) {
    parts.push(h('div', { class: 'cov warn' }, `⚠ ${never.length} module(s) à poids jamais appelé(s) pendant le forward`),
      h('details', {}, h('summary', {}, 'Code mort ou option ignorée ?'),
        ...never.map(m => h('div', {}, h('code', {}, m.module), ` · ×${m.count}`))));
  }
  parts.push(h('p', { class: 'muted small', style: { margin: '6px 0 0' } }, `${a.source || ''} · généré le ${a.generated?.replace('T', ' à ') || '?'}`));
  el.replaceChildren(...parts);
}

// --- chargement ------------------------------------------------------------------------

async function loadArch(name) {
  const data = await api(`arch/${encodeURIComponent(name)}`);
  st.arch = data;
  st.root = data.root;
  st.selected = null;
  st.panelOpen = false;
  prepare(st.root);
  measure(st.root);
  place(st.root, 0, 0);
  nodesLayer.replaceChildren(renderNode(st.root));
  st.root.exp = true;
  st.root.g.classList.add('exp');
  renderSummary();
  drawMinimap();
  fitAll();
  $('#arch-panel').hidden = true;
  store('arch', name);
  await pickRunFor(name);
}

async function pickRunFor(name) {
  const sel = $('#arch-run');
  const remembered = store(`run-${name}`);
  const pref = st.runs.find(r => r.id === remembered)
    || st.runs.find(r => r.name.startsWith(name))
    || null;
  sel.value = pref ? pref.id : '';
  await loadRun(sel.value);
}

async function loadRunsList() {
  const data = await api('runs').catch(() => ({ experiments: [] }));
  st.runs = data.experiments.flatMap(e => e.runs.filter(r => r.has_layers).map(r => ({ ...r, exp: e.name })));
  const sel = $('#arch-run');
  const cur = sel.value;
  sel.replaceChildren(h('option', { value: '' }, st.runs.length ? 'Aucun' : 'Aucun run avec mesures layer/'),
    ...st.runs.map(r => h('option', { value: r.id }, `${r.name}${r.status === 'RUNNING' ? '  (en cours)' : ''}  ·  ${r.exp}`)));
  sel.value = st.runs.some(r => r.id === cur) ? cur : '';
}

function showEmpty(html) {
  const el = $('#arch-empty');
  el.innerHTML = html;
  el.hidden = false;
}

// --- evenements ----------------------------------------------------------------------

function bindEvents() {
  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const r = stage.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0016)));
  }, { passive: false });

  let drag = null;
  svg.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    drag = { x: e.clientX, y: e.clientY, vx: st.view.x, vy: st.view.y, moved: false };
    svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', e => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    drag.moved = true;
    svg.classList.add('dragging');
    setView(st.view.k, drag.vx + dx, drag.vy + dy);
  });
  svg.addEventListener('pointerup', e => {
    svg.classList.remove('dragging');
    const d = drag;
    drag = null;
    if (!d || d.moved) return;
    const target = document.elementFromPoint(e.clientX, e.clientY);
    const hit = target?.closest('.body, .hdr');
    const g = hit?.closest('.n');
    const n = g && st.byId.get(g.dataset.id);
    if (!n) { select(null); return; }
    select(n);
    if (n.children.length && (hit.classList.contains('hdr') || !n.exp)) zoomTo(n);
  });
  svg.addEventListener('dblclick', e => {
    // avec la capture du pointeur, la cible est le svg : on cherche le bloc sous la souris
    const g = document.elementFromPoint(e.clientX, e.clientY)?.closest('.n');
    const n = g && st.byId.get(g.dataset.id);
    if (n?.children.length) zoomTo(n);
  });

  $('#z-in').onclick = () => zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, 1.5);
  $('#z-out').onclick = () => zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, 1 / 1.5);
  $('#z-fit').onclick = () => fitAll(450);
  window.addEventListener('keydown', e => {
    if ($('#tab-arch').hidden || e.target.matches('input, select')) return;
    if (e.key === 'Escape') {
      const f = focusNode();
      let up = f.parent;
      while (up && isGroup(up)) up = up.parent;
      select(null);
      zoomTo(up || st.root);
    } else if (e.key === '0') fitAll(450);
    else if (e.key === '+' || e.key === '=') zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, 1.4);
    else if (e.key === '-') zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, 1 / 1.4);
  });
  new ResizeObserver(() => requestLOD()).observe(stage);

  const mm = $('#minimap');
  let mmDrag = false;
  mm.addEventListener('pointerdown', e => { mmDrag = true; mm.setPointerCapture(e.pointerId); minimapJump(e); });
  mm.addEventListener('pointermove', e => mmDrag && minimapJump(e));
  mm.addEventListener('pointerup', () => { mmDrag = false; });

  const input = $('#arch-search');
  input.addEventListener('input', () => search(input.value));
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter') $('#arch-results button')?.click();
    if (e.key === 'Escape') { input.value = ''; search(''); }
  });
  document.addEventListener('click', e => { if (!e.target.closest('.search')) $('#arch-results').hidden = true; });

  $('#arch-select').addEventListener('change', e => loadArch(e.target.value));
  $('#arch-run').addEventListener('change', e => {
    if (st.arch) store(`run-${st.arch.name}`, e.target.value);
    loadRun(e.target.value);
  });
  $('#arch-metric').addEventListener('change', e => { st.metric = e.target.value; recolor(); refreshPanel(); });
  for (const b of document.querySelectorAll('#arch-scale button')) {
    b.addEventListener('click', () => {
      st.scale = b.dataset.v;
      document.querySelectorAll('#arch-scale button').forEach(x => x.classList.toggle('on', x === b));
      recolor();
    });
  }
  $('#arch-step').addEventListener('input', e => {
    st.si = Number(e.target.value);
    updateStepLabel(); recolor(); refreshPanel();
  });
  $('#arch-play').addEventListener('click', togglePlay);
}

export async function initArch() {
  svg = $('#arch-svg');
  vp = $('#vp');
  nodesLayer = $('#nodes');
  linksLayer = $('#links');
  stage = $('#arch-stage');
  bindEvents();
  await document.fonts.ready;
  const [list] = await Promise.all([api('arch').catch(() => []), loadRunsList()]);
  if (!list.length) {
    showEmpty(`<div><h3>Aucun schéma d'architecture</h3>
      <p>Le dossier <code>visualisation/arch/</code> est vide. Génère-le depuis la racine du projet :</p>
      <pre>python visualisation/build_arch.py</pre>
      <p class="muted">L'onglet Entraînement fonctionne déjà, il lit <code>mlruns/</code>.</p></div>`);
    return;
  }
  // lien direct : ?arch=mai-v2&node=trunk/bloc-3&metric=grad_norm&run=<nom ou id>
  const url = new URLSearchParams(location.search);
  const sel = $('#arch-select');
  sel.replaceChildren(...list.map(a => h('option', { value: a.name }, a.title)));
  const wanted = url.get('arch') || store('arch');
  sel.value = list.some(a => a.name === wanted) ? wanted : list[0].name;
  if (url.get('metric') && STATS[url.get('metric')]) {
    st.metric = url.get('metric');
    $('#arch-metric').value = st.metric;
  }
  await loadArch(sel.value);
  const run = url.get('run') && st.runs.find(r => r.id === url.get('run') || r.name === url.get('run'));
  if (run) { $('#arch-run').value = run.id; await loadRun(run.id); }
  const target = url.get('node') && st.byId.get(url.get('node'));
  if (target) {
    select(target);
    $('#arch-summary').classList.add('folded');
    const v = viewFor(target.children.length ? target : target.parent || target);
    setView(v.k, v.x, v.y);
  }
}

export function archShown() {
  if (st.root) requestLOD();
}

export async function archRefreshRuns() {
  await loadRunsList();
}
