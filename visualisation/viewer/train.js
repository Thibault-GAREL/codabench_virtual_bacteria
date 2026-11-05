// Onglet Entrainement : runs MLflow du projet, courbes superposees, suivi en direct.
//
// Les donnees viennent du store mlruns/ via le serveur. Une couleur par run, dans
// l'ordre fixe de la palette, gardee tant que le run reste selectionne.

import { $, api, h, fmt, seriesColor, cssVar, store } from './common.js';

const GROUP_ORDER = ['loss', 'best', 'général', 'throughput', 'compute', 'router', 'system'];
const TILE_KEYS = ['loss/ce', 'loss', 'loss/val', 'best/loss/val', 'lr', 'grad_norm', 'throughput/tok_s', 'compute/cost_usd'];
const MAX_RUNS = 8;

const st = {
  experiments: [], runs: new Map(),
  slots: new Map(),          // runId -> slot de couleur
  focus: null,
  keys: new Map(),           // runId -> [{key, n, step, last}]
  data: new Map(),           // runId -> Map(key -> {steps, values})
  charts: new Map(),         // key -> {u, el, sig}
  open: new Set(store('open-groups') || ['loss', 'général', 'best']),
  smooth: 0.6, logy: false, auto: true,
  timer: null, tick: 0,
};

const groupOf = key => (key.includes('/') ? key.split('/')[0] : 'général');

// --- runs -----------------------------------------------------------------------

async function loadRuns() {
  const data = await api('runs').catch(() => ({ experiments: [] }));
  st.experiments = data.experiments;
  st.runs.clear();
  for (const e of data.experiments) for (const r of e.runs) st.runs.set(r.id, { ...r, exp: e.name });
  for (const id of [...st.slots.keys()]) if (!st.runs.has(id)) st.slots.delete(id);
  $('#live').hidden = ![...st.runs.values()].some(r => r.status === 'RUNNING' && !r.silent);
}

function renderRunList() {
  const list = $('#run-list');
  const q = $('#run-filter').value.trim().toLowerCase();
  if (!st.experiments.length) {
    list.replaceChildren(h('p', { class: 'nodata' }, 'Aucun run dans mlruns/. Lance un entraînement avec RunLogger, il apparaîtra ici tout seul.'));
    return;
  }
  const blocks = st.experiments.map((e, i) => {
    const runs = e.runs.filter(r => !q || r.name.toLowerCase().includes(q) || e.name.toLowerCase().includes(q));
    if (!runs.length) return null;
    const open = i === 0 || runs.some(r => st.slots.has(r.id)) || !!q;
    return h('details', { open }, h('summary', {}, `${e.name}  ·  ${runs.length}`), ...runs.map(runRow));
  });
  list.replaceChildren(...blocks.filter(Boolean));
}

function runRow(r) {
  const slot = st.slots.get(r.id);
  const on = slot != null;
  const label = r.silent ? 'muet' : { RUNNING: 'en cours', FINISHED: 'terminé', FAILED: 'échec', KILLED: 'arrêté' }[r.status] || r.status;
  const mark = r.silent ? '⚠ ' : { RUNNING: '● ', FINISHED: '✓ ', FAILED: '✕ ', KILLED: '■ ' }[r.status] || '';
  const dur = r.start ? ((r.end || r.last_update || Date.now()) - r.start) / 1000 : null;
  return h('div', {
    class: `run${on ? ' on' : ''}${st.focus === r.id ? ' focus' : ''}`,
    title: 'Clic : afficher ce run. Le carré : le superposer aux autres.',
    onclick: e => {
      if (e.target.closest('.sw')) toggleRun(r.id);
      else focusRun(r.id);
    },
  },
  h('span', { class: 'sw', style: on ? { background: seriesColor(slot) } : {} }),
  h('span', { class: 'nm' }, r.name),
  h('span', { class: `pill ${r.silent ? 'silent' : r.status}` }, mark + label),
  h('span', { class: 'meta' }, `${fmt.ago(r.start)} · ${fmt.dur(dur)}${r.has_layers ? ' · mesures par couche' : ''}`));
}

function toggleRun(id) {
  if (st.slots.has(id)) {
    st.slots.delete(id);
    if (st.focus === id) st.focus = [...st.slots.keys()][0] || null;
  } else {
    const used = new Set(st.slots.values());
    const free = [...Array(MAX_RUNS).keys()].find(i => !used.has(i));
    if (free == null) { alert(`${MAX_RUNS} runs superposés au maximum, au-delà les couleurs ne se distinguent plus.`); return; }
    st.slots.set(id, free);
  }
  store('selected', [...st.slots.keys()]);
  renderRunList();
  refreshAll();
}

function focusRun(id) {
  if (!st.slots.has(id)) {
    // un clic simple remplace la selection, le carre sert a superposer
    st.slots.clear();
    st.slots.set(id, 0);
  }
  st.focus = id;
  store('selected', [...st.slots.keys()]);
  store('focus', id);
  renderRunList();
  refreshAll();
}

// --- donnees ---------------------------------------------------------------------

async function loadKeys(id) {
  const res = await api(`run/${id}/keys`).catch(() => ({ keys: [] }));
  st.keys.set(id, res.keys);
}

async function ensureData(id, keys) {
  if (!keys.length) return;
  if (!st.data.has(id)) st.data.set(id, new Map());
  const cache = st.data.get(id);
  const fresh = keys.filter(k => !cache.has(k));
  const known = st.runs.get(id)?.status === 'RUNNING' ? keys.filter(k => cache.has(k)) : [];
  const ask = async (list, since) => {
    for (let i = 0; i < list.length; i += 40) {
      const part = list.slice(i, i + 40);
      const res = await api(`run/${id}/metrics?since=${since}&keys=${part.map(encodeURIComponent).join(',')}`);
      for (const [k, pts] of Object.entries(res.series)) {
        const cur = cache.get(k) || { steps: [], values: [] };
        const last = cur.steps.at(-1) ?? -Infinity;
        for (const [stp, v] of pts) if (stp > last || since < 0) { cur.steps.push(stp); cur.values.push(v); }
        cache.set(k, cur);
      }
    }
  };
  if (fresh.length) await ask(fresh, -1);
  if (known.length) {
    const since = Math.min(...known.map(k => cache.get(k).steps.at(-1) ?? -1));
    await ask(known, since);
  }
}

function unionKeys() {
  const all = new Map();
  for (const id of st.slots.keys()) for (const k of st.keys.get(id) || []) all.set(k.key, k);
  return [...all.keys()].sort();
}

// --- courbes -----------------------------------------------------------------------

function ema(values, w) {
  if (!w) return values;
  const out = new Array(values.length);
  let acc = 0, n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v == null) { out[i] = null; continue; }
    acc = acc * w + (1 - w) * v;
    n++;
    out[i] = acc / (1 - w ** n);
  }
  return out;
}

function alpha(color, a) {
  if (color.startsWith('#')) {
    const v = color.slice(1);
    const [r, g, b] = [0, 2, 4].map(i => parseInt(v.slice(i, i + 2), 16));
    return `rgba(${r},${g},${b},${a})`;
  }
  return color;
}

function seriesFor(key) {
  const runs = [...st.slots.entries()].filter(([id]) => st.data.get(id)?.has(key));
  const tables = [];
  const series = [{ label: 'step', value: (u, v) => (v == null ? '–' : fmt.int(v)) }];
  const raw = [];
  for (const [id, slot] of runs) {
    const d = st.data.get(id).get(key);
    // tri et dedoublonnage par step (un run repris peut repasser sur des steps)
    const byStep = new Map();
    d.steps.forEach((stp, i) => byStep.set(stp, d.values[i]));
    const xs = [...byStep.keys()].sort((a, b) => a - b);
    let ys = xs.map(x => byStep.get(x));
    if (st.logy) ys = ys.map(v => (v != null && v > 0 ? v : null));
    const color = seriesColor(slot);
    const name = st.runs.get(id)?.name || id.slice(0, 8);
    const smooth = st.smooth > 0 && xs.length > 3;
    if (smooth) {
      tables.push([xs, ys, ema(ys, st.smooth)]);
      raw.push(series.length);
      series.push({ label: `${name} (brut)`, stroke: alpha(color, 0.22), width: 1, spanGaps: true, points: { show: false } });
      series.push({ label: name, stroke: color, width: 2, spanGaps: true, points: { show: xs.length === 1 }, value: (u, v) => fmt.num(v) });
    } else {
      tables.push([xs, ys]);
      series.push({ label: name, stroke: color, width: 2, spanGaps: true, points: { show: xs.length <= 2 }, value: (u, v) => fmt.num(v) });
    }
  }
  if (!tables.length) return null;
  const data = tables.length === 1 ? tables[0] : uPlot.join(tables);
  return { data, series, raw, sig: series.map(x => x.label + x.stroke).join('|') + st.logy };
}

function makeChart(key, card) {
  const spec = seriesFor(key);
  const holder = card.querySelector('.plot');
  const old = st.charts.get(key);
  if (!spec) { old?.u?.destroy(); st.charts.delete(key); holder.replaceChildren(h('p', { class: 'nodata' }, 'Pas de point')); return; }
  // reutilisable seulement si le graphique est bien dans ce cadre (pas un ancien detache)
  if (old && old.sig === spec.sig && holder.contains(old.u.root)) {
    old.u.setData(spec.data, !old.zoomed);
    return;
  }
  old?.u?.destroy();
  holder.replaceChildren();
  const muted = cssVar('--muted'), grid = cssVar('--grid');
  const axis = { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, width: 1, size: 4 }, font: '11px Inter, sans-serif' };
  const entry = { zoomed: false, sig: spec.sig };
  const u = new uPlot({
    width: Math.max(280, holder.clientWidth || 400), height: 210,
    cursor: { sync: { key: 'viz-train' }, points: { size: 7 }, drag: { x: true, y: false } },
    scales: { x: { time: false }, y: { distr: st.logy ? 3 : 1 } },
    axes: [
      { ...axis, values: (_, vals) => vals.map(v => fmt.int(v)) },
      { ...axis, size: 62, values: (_, vals) => vals.map(v => fmt.num(v, 3)) },
    ],
    series: spec.series,
    hooks: { setSelect: [() => { entry.zoomed = true; }] },
  }, spec.data, holder);
  // les courbes brutes n'encombrent pas la legende
  const rows = u.root.querySelectorAll('.u-legend .u-series');
  spec.raw.forEach(i => rows[i] && (rows[i].style.display = 'none'));
  u.over.addEventListener('dblclick', () => { entry.zoomed = false; });
  entry.u = u;
  st.charts.set(key, entry);
}

const resizer = new ResizeObserver(entries => {
  for (const e of entries) {
    const key = e.target.dataset.key;
    const c = st.charts.get(key);
    if (c?.u) c.u.setSize({ width: Math.max(280, e.contentRect.width), height: 210 });
  }
});

function renderGroups() {
  const box = $('#groups');
  if (!st.slots.size) {
    for (const c of st.charts.values()) c.u?.destroy();
    st.charts.clear();
    box.replaceChildren(h('p', { class: 'nodata' }, 'Choisis un run à gauche. Le carré à côté d\'un run le superpose aux autres.'));
    return;
  }
  const q = $('#metric-filter').value.trim().toLowerCase();
  const keys = unionKeys().filter(k => !q || k.toLowerCase().includes(q));
  const groups = new Map();
  for (const k of keys) {
    const g = groupOf(k);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(k);
  }
  const order = [...groups.keys()].sort((a, b) => {
    const ia = GROUP_ORDER.indexOf(a), ib = GROUP_ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const keep = new Set(keys);
  for (const [k, c] of st.charts) if (!keep.has(k)) { c.u?.destroy(); st.charts.delete(k); }
  const blocks = order.map(g => {
    const list = groups.get(g);
    const open = !!q || st.open.has(g);
    const det = h('details', { open, 'data-group': g },
      h('summary', {}, g, h('span', { class: 'muted' }, `${list.length} courbe${list.length > 1 ? 's' : ''}`)));
    const grid = h('div', { class: 'charts' });
    det.append(grid);
    det.addEventListener('toggle', () => {
      if (det.open) st.open.add(g); else st.open.delete(g);
      store('open-groups', [...st.open]);
      // un <details> cree ouvert declenche aussi toggle : ne remplir qu'une fois
      if (det.open && !grid.childElementCount) fillGroup(grid, list);
    });
    if (open) queueMicrotask(() => fillGroup(grid, list));
    return det;
  });
  box.replaceChildren(...(blocks.length ? blocks : [h('p', { class: 'nodata' }, 'Aucune courbe ne correspond au filtre.')]));
}

async function fillGroup(grid, keys) {
  if (!grid.isConnected) return;
  const gen = (grid.gen = (grid.gen || 0) + 1);
  const cards = keys.map(k => h('div', { class: 'chart' }, h('h4', {}, k), h('div', { class: 'plot', 'data-key': k })));
  grid.replaceChildren(...cards);
  for (const k of keys) { st.charts.get(k)?.u?.destroy(); st.charts.delete(k); }
  await Promise.all([...st.slots.keys()].map(id => ensureData(id, keys.filter(k => (st.keys.get(id) || []).some(x => x.key === k)))));
  // un remplissage plus recent du meme groupe a pris la main pendant le chargement
  if (grid.gen !== gen || !grid.isConnected) return;
  cards.forEach((card, i) => {
    makeChart(keys[i], card);
    resizer.observe(card.querySelector('.plot'));
  });
}

async function updateOpenCharts() {
  const open = [...document.querySelectorAll('#groups details[open] .plot')].map(p => p.dataset.key);
  const running = [...st.slots.keys()].filter(id => st.runs.get(id)?.status === 'RUNNING');
  if (!running.length || !open.length) return;
  await Promise.all(running.map(id => ensureData(id, open.filter(k => (st.keys.get(id) || []).some(x => x.key === k)))));
  for (const plot of document.querySelectorAll('#groups details[open] .plot')) {
    makeChart(plot.dataset.key, plot.parentElement);
  }
}

// --- run mis en avant : etat, journal, parametres ---------------------------------------

async function renderStatus() {
  const box = $('#run-status');
  const r = st.focus && st.runs.get(st.focus);
  if (!r) { box.replaceChildren(); return; }
  const stat = await api(`run/${r.id}/status`).catch(() => ({}));
  const keys = st.keys.get(r.id) || [];
  const byKey = new Map(keys.map(k => [k.key, k]));
  const label = r.silent ? 'muet' : { RUNNING: 'en cours', FINISHED: 'terminé', FAILED: 'échec', KILLED: 'arrêté' }[r.status] || r.status;
  const pct = stat.max_steps && stat.step != null ? Math.min(100, (100 * (stat.step + 1)) / stat.max_steps) : null;
  const end = r.end || (r.status === 'RUNNING' ? Date.now() : r.last_update);
  const facts = [
    h('span', {}, 'Expérience ', h('b', {}, r.exp)),
    h('span', {}, 'Démarré ', h('b', {}, r.start ? new Date(r.start).toLocaleString('fr-FR', { dateStyle: 'medium', timeStyle: 'short' }) : '–')),
    h('span', {}, 'Durée ', h('b', {}, fmt.dur(r.start && end ? (end - r.start) / 1000 : null))),
    stat.step != null ? h('span', {}, 'Step ', h('b', {}, fmt.int(stat.step)), stat.max_steps ? ` / ${fmt.int(stat.max_steps)}` : '') : null,
    stat.rate ? h('span', {}, 'Vitesse ', h('b', {}, `${stat.rate.toFixed(stat.rate < 10 ? 2 : 0)} steps/s`)) : null,
    r.status === 'RUNNING' && stat.eta_s != null ? h('span', {}, 'Fin estimée dans ', h('b', {}, fmt.dur(stat.eta_s))) : null,
    h('span', {}, 'Dernière mesure ', h('b', {}, fmt.ago(stat.last_ms))),
  ];
  const tiles = TILE_KEYS.filter(k => byKey.has(k)).slice(0, 6).map(k => h('div', { class: 'tile' },
    h('div', { class: 'k' }, k), h('div', { class: 'v' }, fmt.num(byKey.get(k).last, 4))));
  box.replaceChildren(
    h('h2', {}, r.name, h('span', { class: `pill ${r.silent ? 'silent' : r.status}` }, label)),
    pct != null ? h('div', { class: 'progress', title: `${pct.toFixed(1)} %` }, h('i', { style: { width: `${pct}%` } })) : null,
    h('div', { class: 'facts' }, facts),
    tiles.length ? h('div', { class: 'tiles' }, tiles) : null,
  );
}

async function renderLog() {
  if (!st.focus || $('#bottom-log').hidden) return;
  const res = await api(`run/${st.focus}/log?lines=400`).catch(() => ({ lines: [] }));
  const pre = $('#log');
  const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 8;
  pre.textContent = res.lines.length ? res.lines.join('\n') : 'Pas de run.log trouvé pour ce run (outputs/logs/<run>/ ou artefact MLflow).';
  if (atBottom) pre.scrollTop = pre.scrollHeight;
}

async function renderParams() {
  if (!st.focus) return;
  const res = await api(`run/${st.focus}/params`).catch(() => ({ params: {} }));
  st.params = res.params;
  drawParams();
}
function drawParams() {
  const q = $('#param-filter').value.trim().toLowerCase();
  const rows = Object.entries(st.params || {})
    .filter(([k, v]) => !q || k.toLowerCase().includes(q) || v.toLowerCase().includes(q))
    .map(([k, v]) => h('tr', {}, h('td', {}, k), h('td', {}, v)));
  $('#param-table').replaceChildren(...rows);
}

// --- cycle ------------------------------------------------------------------------------

async function refreshAll() {
  await Promise.all([...st.slots.keys()].map(loadKeys));
  renderGroups();
  await Promise.all([renderStatus(), renderLog(), renderParams()]);
}

async function poll() {
  if (!st.auto || $('#tab-train').hidden) return;
  st.tick++;
  if (st.tick % 3 === 0) { await loadRuns(); renderRunList(); }
  const focus = st.focus && st.runs.get(st.focus);
  const live = [...st.slots.keys()].some(id => st.runs.get(id)?.status === 'RUNNING');
  if (!live) return;
  await Promise.all([...st.slots.keys()].filter(id => st.runs.get(id)?.status === 'RUNNING').map(loadKeys));
  await updateOpenCharts();
  if (focus) await Promise.all([renderStatus(), renderLog()]);
}

export function trainThemeChanged() {
  for (const [key, c] of st.charts) { c.sig = null; const card = c.u?.root?.closest('.chart'); if (card) makeChart(key, card); }
  renderRunList();
}

export async function initTrain() {
  await loadRuns();
  const saved = (store('selected') || []).filter(id => st.runs.has(id)).slice(0, MAX_RUNS);
  saved.forEach((id, i) => st.slots.set(id, i));
  const savedFocus = store('focus');
  st.focus = st.runs.has(savedFocus) && st.slots.has(savedFocus) ? savedFocus : saved[0] || null;
  if (!st.slots.size) {
    const first = st.experiments[0]?.runs[0];
    if (first) { st.slots.set(first.id, 0); st.focus = first.id; }
  }
  renderRunList();

  $('#run-filter').addEventListener('input', renderRunList);
  $('#metric-filter').addEventListener('input', renderGroups);
  $('#smooth').addEventListener('input', e => {
    st.smooth = Number(e.target.value);
    $('#smooth-v').textContent = st.smooth.toFixed(2);
    for (const [key, c] of st.charts) { c.sig = null; const card = c.u?.root?.closest('.chart'); if (card) makeChart(key, card); }
  });
  $('#logy').addEventListener('change', e => { st.logy = e.target.checked; renderGroups(); });
  $('#auto').addEventListener('change', e => { st.auto = e.target.checked; });
  $('#param-filter').addEventListener('input', drawParams);
  for (const b of document.querySelectorAll('#bottom-tabs button')) {
    b.addEventListener('click', () => {
      document.querySelectorAll('#bottom-tabs button').forEach(x => x.classList.toggle('on', x === b));
      $('#bottom-log').hidden = b.dataset.v !== 'log';
      $('#bottom-params').hidden = b.dataset.v !== 'params';
      if (b.dataset.v === 'log') renderLog();
    });
  }
  st.timer = setInterval(poll, 3000);
}

export async function trainShown() {
  await refreshAll();
}
