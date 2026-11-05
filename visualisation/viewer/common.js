// Outils partages par les deux onglets : API, formats, couleurs, petits constructeurs DOM.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export async function api(path) {
  const res = await fetch(`/api/${path}`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`${path} : ${res.status}`);
  return res.json();
}

// Element HTML : h('div', {class: 'x', onclick: f}, enfant, ...)
export function h(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  setAttrs(node, attrs);
  for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid instanceof Node ? kid : String(kid));
  return node;
}

const SVGNS = 'http://www.w3.org/2000/svg';
export function s(tag, attrs = {}, ...kids) {
  const node = document.createElementNS(SVGNS, tag);
  setAttrs(node, attrs);
  for (const kid of kids.flat()) if (kid != null && kid !== false) node.append(kid instanceof Node ? kid : String(kid));
  return node;
}

function setAttrs(node, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else node.setAttribute(k, v === true ? '' : v);
  }
}

export function icon(kind, attrs = {}) {
  return s('use', { href: `#i-${ICONS.has(kind) ? kind : 'group'}`, ...attrs });
}
const ICONS = new Set(['input', 'output', 'embed', 'pos', 'norm', 'attention', 'linear', 'rope', 'sdpa',
  'softmax', 'act', 'ffn', 'moe', 'router', 'expert', 'head', 'loss', 'mtp', 'block', 'stack', 'cache',
  'latent', 'group', 'op']);

// --- nombres ---------------------------------------------------------------

export const fmt = {
  num(v, digits = 4) {
    if (v == null || Number.isNaN(v)) return '–';
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e6) return compact(v);
    if (a >= 1000) return Math.round(v).toLocaleString('fr-FR');
    if (a < 1e-3) return v.toExponential(2);
    return Number(v.toPrecision(digits)).toString();
  },
  params(n) {
    if (n == null) return '–';
    if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e8 ? 1 : 2)} M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)} k`;
    return String(n);
  },
  int(n) { return n == null ? '–' : Math.round(n).toLocaleString('fr-FR'); },
  dur(sec) {
    if (sec == null || !Number.isFinite(sec)) return '–';
    sec = Math.max(0, Math.round(sec));
    const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
    if (hh) return `${hh} h ${String(mm).padStart(2, '0')}`;
    if (mm) return `${mm} min ${String(ss).padStart(2, '0')}`;
    return `${ss} s`;
  },
  ago(ms) {
    if (!ms) return '–';
    const sec = (Date.now() - ms) / 1000;
    if (sec < 60) return `il y a ${Math.max(1, Math.round(sec))} s`;
    if (sec < 3600) return `il y a ${Math.round(sec / 60)} min`;
    if (sec < 86400) return `il y a ${Math.round(sec / 3600)} h`;
    return new Date(ms).toLocaleDateString('fr-FR', { day: 'numeric', month: 'short', year: 'numeric' });
  },
};

function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)} G`;
  return `${(v / 1e6).toFixed(2)} M`;
}

// --- couleurs ----------------------------------------------------------------

// Palette categorielle validee (skill dataviz), dans son ordre fixe : une couleur par run.
export function seriesColor(slot) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--s${(slot % 8) + 1}`).trim();
}
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Rampe sequentielle bleue (une seule teinte, clair vers fonce).
export const RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
  '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];
export function rampColor(t) {
  const i = Math.max(0, Math.min(RAMP.length - 1, Math.round(t * (RAMP.length - 1))));
  return { color: RAMP[i], dark: i >= 7 };
}

// Divergente bleu, gris neutre, rouge : sous-charge, equilibre, surcharge.
export function divergingColor(t) {
  // t dans [-1, 1]
  const mid = hexToRgb(cssVar('--div-mid') || '#f0efec');
  const pole = hexToRgb(t < 0 ? '#2a78d6' : '#e34948');
  const a = Math.min(1, Math.abs(t));
  const c = mid.map((m, i) => Math.round(m + (pole[i] - m) * a));
  return { color: `rgb(${c.join(',')})`, dark: a > 0.6 };
}
function hexToRgb(hex) {
  const v = hex.replace('#', '');
  return [0, 2, 4].map(i => parseInt(v.slice(i, i + 2), 16));
}

// Familles de couleur des blocs (theme "Paper" de schema-svg).
export const FAMILY = {
  input: '#F97316', embed: '#06B6D4', pos: '#06B6D4',
  attention: '#3B82F6', sdpa: '#3B82F6', rope: '#3B82F6', cache: '#3B82F6', latent: '#3B82F6',
  ffn: '#8B5CF6', expert: '#8B5CF6',
  moe: '#EC4899', router: '#EC4899',
  norm: '#64748B', block: '#64748B', stack: '#64748B', group: '#64748B',
  head: '#10B981', output: '#10B981',
  loss: '#EF4444', mtp: '#EAB308',
};
// ces familles prennent la couleur du bloc parent (une projection dans l'attention reste bleue)
export const INHERIT = new Set(['linear', 'act', 'softmax', 'op', 'add', 'mul', 'concat', 'split']);

export const KIND_LABEL = {
  input: 'Entrée', output: 'Sortie', embed: 'Embedding', pos: 'Position', norm: 'Normalisation',
  attention: 'Attention', linear: 'Projection linéaire', rope: 'Rotation RoPE', sdpa: 'Produit scalaire',
  softmax: 'Softmax', act: 'Activation', ffn: 'Feed-forward', moe: 'Mixture of Experts', router: 'Routeur',
  expert: 'Expert', head: 'Tête de sortie', loss: 'Perte', mtp: 'Multi-Token Prediction',
  block: 'Bloc', stack: 'Pile de couches', cache: 'Cache', latent: 'Latent', group: 'Groupe',
  add: 'Addition résiduelle', mul: 'Produit terme à terme', concat: 'Concaténation', split: 'Séparation', op: 'Opération',
};

// --- theme -------------------------------------------------------------------

export function initTheme(button, onChange) {
  const order = ['auto', 'light', 'dark'];
  const names = { auto: 'Auto', light: 'Clair', dark: 'Sombre' };
  let mode = 'auto';
  try { mode = localStorage.getItem('viz-theme') || 'auto'; } catch { /* stockage indisponible */ }
  const apply = () => {
    if (mode === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', mode);
    button.textContent = names[mode];
    onChange?.();
  };
  button.addEventListener('click', () => {
    mode = order[(order.indexOf(mode) + 1) % order.length];
    try { localStorage.setItem('viz-theme', mode); } catch { /* idem */ }
    apply();
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => mode === 'auto' && onChange?.());
  apply();
}

export function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(`viz-${key}`));
    localStorage.setItem(`viz-${key}`, JSON.stringify(value));
  } catch { return null; }
  return value;
}
