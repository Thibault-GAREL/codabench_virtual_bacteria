// Point d'entree : onglets, theme, nom du projet.

import { $, api, initTheme, store } from './common.js';
import { initArch, archShown } from './arch.js';
import { initTrain, trainShown, trainThemeChanged } from './train.js';

let trainReady = null;

async function showTab(name) {
  for (const b of document.querySelectorAll('.tabs button')) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  $('#tab-arch').hidden = name !== 'arch';
  $('#tab-train').hidden = name !== 'train';
  store('tab', name);
  if (name === 'arch') archShown();
  else {
    trainReady = trainReady || initTrain();
    await trainReady;
    await trainShown();
  }
}

async function main() {
  initTheme($('#theme'), () => trainReady && trainThemeChanged());
  const info = await api('info').catch(() => null);
  if (info) {
    $('#project').textContent = info.project;
    document.title = `Model viz · ${info.project}`;
  }
  for (const b of document.querySelectorAll('.tabs button')) b.addEventListener('click', () => showTab(b.dataset.tab));
  // un lien vers un bloc (?arch= ou ?node=) ouvre toujours l'onglet Architecture
  const url = new URLSearchParams(location.search);
  const initial = url.get('tab') || (url.has('arch') || url.has('node') ? 'arch' : store('tab')) || 'arch';
  await initArch().catch(err => console.error(err));
  await showTab(initial === 'train' ? 'train' : 'arch');
}

main();
