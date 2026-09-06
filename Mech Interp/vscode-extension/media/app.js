'use strict';

const host = acquireVsCodeApi();
const config = JSON.parse(document.getElementById('config').textContent);
const app = document.getElementById('app');
const status = document.getElementById('status');
const pending = new Map(), notes = new Map();
let serial = 0, data, currentTab = host.getState()?.tab || config.tab || 'continue', paper, paperLoading = false;
let arenaChapter = 'all', arenaKind = 'exercise', arenaSearch = '', libraryModule = 'all';
const tabs = [['continue', 'Continue'], ['plan', 'Study plan'], ['course', 'Course checklists'], ['arena', 'ARENA'], ['library', 'Reading library'], ['work', 'Your work']];
const scrolls = host.getState()?.scrolls || {};
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const openButton = (target, label, primary = false) => `<button class="${primary ? 'primary' : 'link-button'}" data-open="${esc(target)}">${esc(label)}</button>`;
const noteBox = (key, label, placeholder = '') => `<label class="note-label">${esc(label)}<textarea data-note="${esc(key)}" placeholder="${esc(placeholder)}">${esc(data.progress.notes?.[key] || '')}</textarea></label><p class="muted small">Saves automatically. Save notebook code and written answers with Ctrl+S.</p>`;

function api(action, payload = {}) {
  const requestId = ++serial;
  return new Promise((resolve, reject) => {
    pending.set(requestId, { resolve, reject });
    host.postMessage({ requestId, action, payload });
  });
}

function report(message, error = false) {
  status.textContent = message;
  status.className = error ? 'error' : '';
}

window.addEventListener('message', event => {
  const message = event.data;
  if (message.type === 'navigate') { navigate(message.tab).catch(error => report(error.message, true)); return; }
  const waiting = pending.get(message.requestId);
  if (!waiting) return;
  pending.delete(message.requestId);
  if (message.error) waiting.reject(new Error(message.error)); else waiting.resolve(message.result);
});

async function save(action, payload) {
  report('Saving…');
  try {
    const result = await api(action, payload);
    if (result.progress && data) data.progress = result.progress;
    report('Saved locally.');
    return result;
  } catch (error) { report(`Not saved: ${error.message}`, true); throw error; }
}

function queueNote(key, value) {
  const previous = notes.get(key);
  if (previous?.timer) clearTimeout(previous.timer);
  const item = { value, timer: undefined, promise: undefined };
  notes.set(key, item);
  report('Saving…');
  item.timer = setTimeout(() => persistNote(key, item).catch(() => {}), 500);
}

async function persistNote(key, item) {
  clearTimeout(item.timer); item.timer = undefined;
  if (!item.promise) item.promise = save('note', { key, value: item.value });
  try { await item.promise; }
  catch (error) { item.promise = undefined; throw error; }
  if (notes.get(key) === item) notes.delete(key);
}

async function flushNotes() { await Promise.all([...notes].map(([key, item]) => persistNote(key, item))); }

function chapterLabel(chapter) {
  const labels = { chapter0_fundamentals: '0 · Fundamentals', chapter1_transformer_interp: '1 · Transformer interpretability',
    chapter2_rl: '2 · Reinforcement learning', chapter3_llm_evals: '3 · LLM evaluations', chapter4_alignment_science: '4 · Alignment science' };
  return labels[chapter] || chapter.replaceAll('_', ' ');
}

function resourceCard(resource) {
  let actions = '';
  if (resource.notebook) actions += openButton(resource.notebook, 'Open exercises', true);
  if (resource.local) actions += `<button class="primary" data-paper="${esc(resource.id)}">Read paper</button>`;
  actions += openButton(resource.url, 'Original source');
  if (resource.web) actions += openButton(resource.web, 'Web edition');
  const options = data.readingStates.map(value => `<option value="${esc(value)}" ${value === (data.progress.reading?.[resource.id] || 'Not started') ? 'selected' : ''}>${esc(value)}</option>`).join('');
  return `<article class="card"><div class="eyebrow">${esc(resource.type)} · ${esc(resource.author)}</div><h2>${esc(resource.title)}</h2>
    <p>${esc(resource.target)}</p><p class="muted">${esc(resource.depth)} · ${esc(resource.note)}</p><div class="actions">${actions}</div>
    <label class="inline-label">Reading status <select data-reading="${esc(resource.id)}">${options}</select></label></article>`;
}

function continuePage() {
  const total = data.course.modules.reduce((sum, module) => sum + module.tasks.length, 0);
  const count = data.course.modules.reduce((sum, module) => sum + module.tasks.filter(t => (data.progress.completed?.[module.id] || []).includes(t.id)).length, 0);
  const next = data.course.modules.map(m => [m, m.tasks.find(t => !(data.progress.completed?.[m.id] || []).includes(t.id))]).find(([, t]) => t);
  const current = data.course.modules.filter(module => Number(module.id) >= 3).find(module => module.tasks.some(task => !(data.progress.completed?.[module.id] || []).includes(task.id)));
  const exercise = current && data.course.resources.find(resource => resource.module === current.id && resource.notebook);
  const nextAction = exercise ? openButton(exercise.notebook, `Open ${exercise.title}`, true)
    : current ? openButton(current.file, 'Open research scaffold', true) : '<button class="primary" data-create="experiment">New experiment</button>';
  return `<section class="hero"><div class="eyebrow">${current ? `Continue · Module ${current.id}` : 'Your next research question'}</div><h1>${esc(current?.title || 'Choose your next experiment')}</h1>
    <p>${esc(current?.subtitle || 'Choose a bounded question, inspect prior work, and run a small pilot.')}</p>
    <p class="muted small">The core sequence advances with your course checklists. The study plan pairs each stage with readings and experiments.</p>
    <div class="actions">${nextAction}<button class="link-button" data-tab="plan">Continue your study plan →</button>${current ? openButton(current.file, 'Module notes') : ''}</div></section>
    <div class="two-col"><section class="card"><h2>Saved checklist</h2><p><strong>${count} / ${total}</strong> tasks checked.</p>
    ${next ? `<p class="muted">Next unchecked item: Module ${esc(next[0].id)} · ${esc(next[1].label)}.</p>` : '<p>The course checklist is complete.</p>'}
    <button class="link-button" data-tab="course">Review checklists</button></section>
    <section class="card"><h2>Your workspace</h2><p>All five ARENA chapters, local paper readers, and fresh experiment notebooks.</p>
    <div class="actions"><button class="link-button" data-tab="arena">Browse ARENA</button><button class="link-button" data-tab="work">Start an experiment</button>${openButton('arena/00_Environment_Check.ipynb', 'Check notebook setup')}</div></section></div>
    ${noteBox('session', 'A note for your next session', 'What did you learn? Where will you pick up?')}`;
}

function coursePage() {
  return `<h1>Course checklists</h1><p class="muted">These checkmarks and notes are shared with JupyterLab. Tick a task when you can explain it.</p>` + data.course.modules.map(module => {
    const done = data.progress.completed?.[module.id] || [];
    return `<article class="card"><div class="eyebrow">Module ${esc(module.id)} · ${module.tasks.filter(t => done.includes(t.id)).length}/${module.tasks.length} checked</div>
      <h2>${esc(module.title)}</h2><p>${esc(module.subtitle)}</p>${openButton(module.file, 'Open module notebook', true)}
      <div class="checklist">${module.tasks.map(task => `<label><input type="checkbox" data-module="${esc(module.id)}" data-task="${esc(task.id)}" ${done.includes(task.id) ? 'checked' : ''}><span>${esc(task.label)}</span></label>`).join('')}</div>
      ${noteBox('module-' + module.id, 'Next time, start here')}
      <details><summary>Readings and exercises for this module</summary>${data.course.resources.filter(r => r.module === module.id).map(resourceCard).join('')}</details></article>`;
  }).join('');
}

function arenaPage() {
  const chapters = [...new Set(data.arena.map(record => record.chapter))];
  return `<h1>ARENA notebooks</h1><p>Open an exercise as a native VS Code notebook. Its existing Python environment is selected for you.</p>
    <p class="muted">Setup/import checks passed for all 75 imported notebooks. Full-run coverage and later API, model-access, and Linux requirements are listed in ${openButton('arena/VALIDATION.md', 'Validation details')}.</p>
    <div class="filters"><label>Chapter<select id="arena-chapter"><option value="all">All chapters</option>${chapters.map(ch => `<option value="${esc(ch)}" ${arenaChapter === ch ? 'selected' : ''}>${esc(chapterLabel(ch))}</option>`).join('')}</select></label>
    <label>Show<select id="arena-kind">${[['exercise', 'Exercises'], ['solution', 'Reference solutions — contain answers'], ['additional', 'Additional training notebooks']].map(([value, label]) => `<option value="${value}" ${arenaKind === value ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
    <label>Find a notebook<input id="arena-search" type="search" placeholder="Induction, PPO, superposition…" value="${esc(arenaSearch)}"></label></div><div id="arena-list"></div>`;
}

function renderArenaList() {
  const rows = data.arena.filter(r => r.kind === arenaKind && (arenaChapter === 'all' || arenaChapter === r.chapter)
    && (r.title + ' ' + r.local).toLowerCase().includes(arenaSearch.toLowerCase()));
  document.getElementById('arena-list').innerHTML = `<p class="muted">${rows.length} notebooks</p>` + rows.map(record => {
    const title = record.title.replace(/ (exercises|solutions)$/, '');
    const kernel = record.local.includes('/1.4.2_') ? 'ARENA (SAE circuits)' : 'ARENA (local GPU)';
    return `<article class="notebook-row"><div><h2>${openButton(record.local, title)}</h2><p class="muted small">${esc(chapterLabel(record.chapter))} · ${kernel}</p></div>${openButton(record.local, 'Open →')}</article>`;
  }).join('');
}

function libraryPage() {
  return `<h1>Reading library</h1><p>Read the assigned sections for your current task. Papers and web sources open inside VS Code.</p>
    <label class="inline-label">Show<select id="library-module"><option value="all">All resources</option>${data.course.modules.map(m => `<option value="${m.id}" ${libraryModule === m.id ? 'selected' : ''}>${m.id} · ${esc(m.title)}</option>`).join('')}<option value="later" ${libraryModule === 'later' ? 'selected' : ''}>Later reading</option></select></label>
    ${data.course.resources.filter(r => libraryModule === 'all' || r.module === libraryModule).map(resourceCard).join('')}`;
}

function workPage() {
  return `<h1>Your work</h1><p>Create a notebook for a new experiment or paper. Your existing work stays in this same workspace.</p>
    <div class="actions"><button class="primary" data-create="experiment">New experiment</button><button data-create="paper">New paper note</button></div>
    <ul class="work-list">${[...(data.progress.created || [])].reverse().map(file => `<li>${openButton(file, file.split('/').pop())}</li>`).join('') || '<li class="muted">Your new notebooks will appear here.</li>'}</ul>
    ${noteBox('questions', 'Questions to return to', 'Capture an idea without interrupting your current task.')}
    <section class="card"><h2>Save another resource</h2><form id="bookmark-form"><label>Title<input name="title" required></label><label>URL<input name="url" type="url" required placeholder="https://…"></label><label>When will it help?<input name="reason"></label><button type="submit">Save resource</button></form>
    <ul>${(data.progress.bookmarks || []).map(bookmark => `<li>${openButton(bookmark.url, bookmark.title)}<p class="muted small">${esc(bookmark.reason)}</p></li>`).join('')}</ul></section>`;
}

function renderDashboard() {
  host.setState({ tab: currentTab, scrolls });
  const header = `<header><div><span class="eyebrow">Mech Interp Workbench</span><p class="small muted">Your course, reading, and experiments</p></div><button id="refresh">Refresh</button></header>
    <nav aria-label="Learning dashboard">${tabs.map(([id, title]) => `<button data-tab="${id}" class="${currentTab === id ? 'selected' : ''}" aria-current="${currentTab === id ? 'page' : 'false'}">${title}</button>`).join('')}</nav>`;
  let body;
  if (currentTab === 'plan') body = `<div class="reader-actions"><button data-tab="course">Open course checklists</button>${openButton('LEARNING_PLAN.md', 'Open plan beside this tab')}</div><article class="markdown" data-base="${esc(data.plan.path)}">${data.plan.html}</article>`;
  else body = ({ continue: continuePage, course: coursePage, arena: arenaPage, library: libraryPage, work: workPage }[currentTab] || continuePage)();
  app.innerHTML = `${header}<main>${body}</main>`;
  if (currentTab === 'arena') renderArenaList();
}

async function navigate(tab) {
  if (config.mode !== 'dashboard') return api('dashboard', { tab });
  await flushNotes();
  scrolls[currentTab] = Number(window.scrollY) || 0;
  currentTab = tabs.some(([id]) => id === tab) ? tab : 'continue';
  data = await api('snapshot');
  renderDashboard();
  window.scrollTo(0, scrolls[currentTab] || 0);
}

async function renderPaper(request = {}) {
  if (paperLoading) return;
  paperLoading = true;
  app.querySelectorAll('.paper-controls button, .paper-controls input, .paper-controls select').forEach(input => { input.disabled = true; });
  report('Opening paper…');
  try {
  paper = await api('paper', { id: config.id, mode: request.mode || paper?.mode || 'Page', ...request });
  paper.mode = request.mode || (paper.image ? 'Page' : 'Text');
  const resource = paper.resource;
  app.innerHTML = `<header><button data-home>← Learning dashboard</button>${openButton(resource.url, 'Original source')}</header>
    <main class="paper"><div class="eyebrow">${esc(resource.author)}</div><h1>${esc(resource.title)}</h1><p>${esc(resource.target)}</p>
    <div class="paper-controls"><button data-page="${paper.page - 1}" ${paper.page === 1 ? 'disabled' : ''}>← Previous</button>
    <label>Page<input id="paper-page" type="number" min="1" max="${paper.count}" value="${paper.page}"></label><span>of ${paper.count}</span>
    <button data-page="${paper.page + 1}" ${paper.page === paper.count ? 'disabled' : ''}>Next →</button>
    <label>View<select id="paper-mode"><option value="Page" ${paper.mode === 'Page' ? 'selected' : ''}>Page</option><option value="Text" ${paper.mode === 'Text' ? 'selected' : ''}>Text</option></select></label></div>
    <p class="muted small">Your page is saved when you move.</p>
    ${paper.image ? `<img class="paper-image" src="data:image/png;base64,${paper.image}" alt="Page ${paper.page} of ${esc(resource.title)}">` : `<pre class="paper-text">${esc(paper.text)}</pre>`}</main>`;
  report(`Page ${paper.page} of ${paper.count}.`);
  } finally {
    paperLoading = false;
    app.querySelectorAll('.paper-controls input, .paper-controls select').forEach(input => { input.disabled = false; });
    app.querySelectorAll('[data-page]').forEach(button => { button.disabled = Number(button.dataset.page) < 1 || Number(button.dataset.page) > paper.count; });
  }
}

async function act(event) {
  const button = event.target.closest('button');
  const link = event.target.closest('a');
  if (link) {
    event.preventDefault();
    const href = link.getAttribute('href');
    if (href?.startsWith('#')) { document.getElementById(href.slice(1))?.scrollIntoView(); return; }
    if (href) { await flushNotes(); await api('open', { target: href, base: link.closest('[data-base]')?.dataset.base || '' }); }
    return;
  }
  if (!button || (button.type === 'submit' && button.closest('form'))) return;
  if (button.dataset.tab) return navigate(button.dataset.tab);
  if (button.id === 'refresh') return navigate(currentTab);
  await flushNotes();
  button.disabled = true;
  try {
    if (button.dataset.open) await api('open', { target: button.dataset.open });
    else if (button.dataset.paper) await api('paperOpen', { id: button.dataset.paper });
    else if (button.dataset.create) { const result = await save('create', { kind: button.dataset.create }); data.progress = result.progress; renderDashboard(); }
    else if (button.hasAttribute('data-home')) await api('dashboard', {});
    else if (button.dataset.edit) await api('edit', { path: button.dataset.edit });
    else if (button.hasAttribute('data-page')) await renderPaper({ page: Number(button.dataset.page), mode: paper.mode });
  } finally { button.disabled = false; }
}

app.addEventListener('click', event => act(event).catch(error => report(error.message, true)));
app.addEventListener('input', event => {
  if (event.target.dataset.note) queueNote(event.target.dataset.note, event.target.value);
  if (event.target.id === 'arena-search') { arenaSearch = event.target.value; renderArenaList(); }
});
app.addEventListener('change', event => {
  const input = event.target;
  (async () => {
    if (input.dataset.task) {
      const before = (data.progress.completed?.[input.dataset.module] || []).includes(input.dataset.task);
      try { await save('complete', { module: input.dataset.module, task: input.dataset.task, value: input.checked }); }
      catch (error) { input.checked = before; throw error; }
    } else if (input.dataset.reading) {
      const before = data.progress.reading?.[input.dataset.reading] || 'Not started';
      try { await save('reading', { id: input.dataset.reading, value: input.value }); }
      catch (error) { input.value = before; throw error; }
    }
    else if (input.id === 'arena-chapter') { arenaChapter = input.value; renderArenaList(); }
    else if (input.id === 'arena-kind') { arenaKind = input.value; renderArenaList(); }
    else if (input.id === 'library-module') { await flushNotes(); libraryModule = input.value; renderDashboard(); }
    else if (input.id === 'paper-page') await renderPaper({ page: Number(input.value), mode: paper.mode });
    else if (input.id === 'paper-mode') await renderPaper({ page: paper.page, mode: input.value });
  })().catch(error => report(error.message, true));
});
app.addEventListener('submit', event => {
  event.preventDefault();
  if (event.target.id !== 'bookmark-form') return;
  const fields = new FormData(event.target);
  save('bookmark', Object.fromEntries(fields.entries())).then(renderDashboard).catch(() => {});
});

(async () => {
  if (config.mode === 'paper') await renderPaper();
  else if (config.mode === 'document') {
    const doc = await api('document', { path: config.path });
    app.innerHTML = `<header><button data-home>← Learning dashboard</button><button data-edit="${esc(doc.path)}">Edit Markdown</button></header><main class="markdown" data-base="${esc(doc.path)}">${doc.html}</main>`;
  } else { data = await api('snapshot'); renderDashboard(); window.scrollTo(0, scrolls[currentTab] || 0); }
})().catch(error => report(error.message, true));
