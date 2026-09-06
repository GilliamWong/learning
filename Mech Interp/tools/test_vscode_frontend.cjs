// Exercise the real dashboard script in a DOM, without controlling VS Code.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { spawn } = require('node:child_process');
const { parseHTML } = require('../.runtime/vscode-unit/node_modules/linkedom');

const root = path.resolve(__dirname, '..');
const fixture = fs.mkdtempSync(path.join(root, '.runtime', 'vscode-frontend-'));
for (const filename of ['curriculum.json', 'LEARNING_PLAN.md']) fs.copyFileSync(path.join(root, filename), path.join(fixture, filename));
fs.cpSync(path.join(root, 'templates'), path.join(fixture, 'templates'), { recursive: true });
fs.mkdirSync(path.join(fixture, 'arena'));
fs.copyFileSync(path.join(root, 'arena/manifest.json'), path.join(fixture, 'arena/manifest.json'));
const localPaper = JSON.parse(fs.readFileSync(path.join(root, 'curriculum.json'), 'utf8')).resources.find(resource => resource.id === 'induction').local;
fs.mkdirSync(path.dirname(path.join(fixture, localPaper)), { recursive: true });
fs.copyFileSync(path.join(root, localPaper), path.join(fixture, localPaper));
const calls = [];
let failNextSave = false;

function backend(action, payload) {
  calls.push({ action, payload });
  if (['open', 'dashboard', 'paperOpen'].includes(action)) return Promise.resolve({});
  if (failNextSave && action === 'note') { failNextSave = false; return Promise.reject(new Error('Simulated temporary save failure')); }
  return new Promise((resolve, reject) => {
    const child = spawn(path.join(root, '.venv/Scripts/python.exe'), ['-X', 'utf8', path.join(root, 'tools/vscode_bridge.py'), action, '--root', fixture], { cwd: root, windowsHide: true });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk.toString(); }); child.stderr.on('data', chunk => { err += chunk.toString(); });
    child.on('error', reject);
    child.on('close', () => { try { const result = JSON.parse(out); result.ok ? resolve(result.result) : reject(new Error(result.error)); } catch (error) { reject(new Error(err || error.message)); } });
    child.stdin.end(JSON.stringify(payload));
  });
}

const { window, document } = parseHTML('<html><body><div id="app"></div><div id="status"></div><script id="config" type="application/json">{"mode":"dashboard","tab":"continue"}</script></body></html>');
// Match the browser default: a button's type is "submit" even outside a form.
Object.defineProperty(window.HTMLButtonElement.prototype, 'type', { get() { return this.getAttribute('type') || 'submit'; } });
window.scrollTo = () => {};
const native = { getState: () => undefined, setState: () => {}, postMessage: message => {
  backend(message.action, message.payload).then(result => respond({ requestId: message.requestId, result }), error => respond({ requestId: message.requestId, error: error.message }));
} };
function respond(data) { const event = new window.Event('message'); event.data = data; window.dispatchEvent(event); }
class FormDataForDOM {
  constructor(form) { this.values = [...form.querySelectorAll('[name]')].map(input => [input.getAttribute('name'), input.value]); }
  entries() { return this.values[Symbol.iterator](); }
}
const context = vm.createContext({ window, document, acquireVsCodeApi: () => native, setTimeout, clearTimeout, console,
  URL, URLSearchParams, FormData: FormDataForDOM, Map, Promise, Date });
vm.runInContext(fs.readFileSync(path.join(root, 'vscode-extension/media/app.js'), 'utf8'), context);
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, label) { for (let i = 0; i < 500; i++) { if (predicate()) return; await pause(10); } throw new Error(`Timed out: ${label}. ${document.getElementById('status').textContent}`); }
function click(selector) { const element = document.querySelector(selector); assert.ok(element, selector); element.dispatchEvent(new window.Event('click', { bubbles: true, cancelable: true })); }
function change(element) { element.dispatchEvent(new window.Event('change', { bubbles: true })); }
function select(id, value) {
  const element = document.getElementById(id), options = [...element.querySelectorAll('option')];
  // linkedom clears the selection when any option is deselected, so select the target last.
  for (const option of options) option.selected = false;
  options.find(option => option.getAttribute('value') === value).selected = true;
  change(element);
}
const saved = () => JSON.parse(fs.readFileSync(path.join(fixture, 'progress.json'), 'utf8'));

(async () => {
  await until(() => document.querySelector('button[data-tab="plan"]'), 'dashboard startup');
  click('.hero button[data-tab="plan"]');
  await until(() => document.querySelector('.markdown'), 'study plan navigation');
  const planLink = [...document.querySelectorAll('.markdown a')].find(link => link.textContent === 'ARENA 1.1 exercise notebook');
  assert.ok(planLink);
  planLink.dispatchEvent(new window.Event('click', { bubbles: true, cancelable: true }));
  await until(() => calls.some(c => c.action === 'open' && c.payload.target.includes('1.1_')), 'plan notebook link');
  assert.equal(calls.find(c => c.action === 'open').payload.base, 'LEARNING_PLAN.md');

  click('nav button[data-tab="course"]');
  await until(() => document.querySelector('[data-task="batch"]'), 'course checklists');
  const checkbox = document.querySelector('[data-task="batch"]'); checkbox.checked = true; change(checkbox);
  await until(() => fs.existsSync(path.join(fixture, 'progress.json')) && saved().completed?.['01']?.includes('batch'), 'checkbox saved');
  const note = document.querySelector('[data-note="module-01"]'); note.value = 'My next step: trace dimensions ✓';
  note.dispatchEvent(new window.Event('input', { bubbles: true }));
  click('nav button[data-tab="arena"]');
  await until(() => document.getElementById('arena-list'), 'leaving a note saves first');
  assert.equal(saved().notes['module-01'], 'My next step: trace dimensions ✓');
  assert.equal(document.querySelectorAll('.notebook-row').length, 34);
  select('arena-chapter', 'chapter2_rl');
  assert.equal(document.querySelectorAll('.notebook-row').length, 6);
  const search = document.getElementById('arena-search'); search.value = 'PPO'; search.dispatchEvent(new window.Event('input', { bubbles: true }));
  assert.equal(document.querySelectorAll('.notebook-row').length, 1);
  assert.match(document.querySelector('.notebook-row').textContent, /PPO/);

  click('nav button[data-tab="library"]');
  await until(() => document.querySelector('[data-paper="induction"]'), 'reading library');
  click('[data-paper="induction"]');
  await until(() => calls.some(c => c.action === 'paperOpen' && c.payload.id === 'induction'), 'local paper link');

  click('nav button[data-tab="work"]');
  await until(() => document.querySelector('[data-create="experiment"]'), 'your work');
  click('[data-create="experiment"]');
  await until(() => saved().created?.length === 1, 'experiment creation');
  assert.equal(JSON.parse(fs.readFileSync(path.join(fixture, saved().created[0]), 'utf8')).metadata.kernelspec.name, 'arena');

  click('nav button[data-tab="continue"]');
  await until(() => document.querySelector('[data-note="session"]'), 'session note');
  failNextSave = true;
  const session = document.querySelector('[data-note="session"]'); session.value = 'Keep this when a save fails';
  session.dispatchEvent(new window.Event('input', { bubbles: true }));
  await until(() => document.getElementById('status').textContent.includes('Not saved'), 'save error surfaced');
  assert.equal(session.value, 'Keep this when a save fails');
  click('#refresh');
  await until(() => saved().notes?.session === 'Keep this when a save fails', 'retry the failed save');

  click('nav button[data-tab="course"]');
  await until(() => document.querySelector('[data-module="03"]'), 'module 3 tasks');
  const tasks = [...document.querySelectorAll('[data-module="03"]')];
  for (const task of tasks) { task.checked = true; change(task); }
  await until(() => saved().completed?.['03']?.length === tasks.length, 'module 3 completion');
  click('nav button[data-tab="continue"]');
  await until(() => document.querySelector('.hero h1')?.textContent === 'Find an induction mechanism', 'Continue advances to induction');
  assert.ok(document.querySelector('.hero button[data-open*="1.2_"]'));

  await vm.runInContext('config.mode = "paper"; config.id = "induction"; renderPaper({mode: "Text"})', context);
  assert.ok(document.querySelector('.paper-text').textContent.length > 100);
  click('[data-page="2"]');
  await until(() => document.getElementById('paper-page')?.value === '2', 'paper pagination');
  assert.equal(saved().notes['paper-page-induction'], '2');
  const mode = document.getElementById('paper-mode');
  for (const option of mode.querySelectorAll('option')) option.selected = false;
  [...mode.querySelectorAll('option')].find(option => option.textContent === 'Page').selected = true;
  change(mode);
  await until(() => document.querySelector('.paper-image'), 'paper page image');
  const image = Buffer.from(document.querySelector('.paper-image').getAttribute('src').split(',')[1], 'base64');
  assert.equal(image.subarray(1, 4).toString(), 'PNG');
  console.log(JSON.stringify({ passed: true, checks: ['dashboard navigation', 'plan links', 'checklists persist', 'notes flush before navigation', 'ARENA filtering', 'paper links', 'new experiment copy', 'failed save preserves text and can retry', 'Continue advances with checklists', 'paper pagination, text view and image view'] }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  await pause(200);
  // Only the temporary test fixture created above is removed.
  assert.ok(path.resolve(fixture).startsWith(path.join(root, '.runtime', 'vscode-frontend-')));
  fs.rmSync(fixture, { recursive: true });
});
