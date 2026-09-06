'use strict';

const vscode = require('vscode');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');

let context, root, dashboard, channel;
const panels = new Map();
let writes = Promise.resolve();
let jupyter;

function workspaceRoot() {
  return vscode.workspace.workspaceFolders?.map(f => f.uri.fsPath).find(folder =>
    fs.existsSync(path.join(folder, 'curriculum.json')) && fs.existsSync(path.join(folder, 'tools', 'vscode_bridge.py')));
}

function workspaceFile(relative, base = '') {
  const target = path.resolve(root, base ? path.dirname(base) : '', relative);
  const real = fs.realpathSync(target);
  const within = path.relative(fs.realpathSync(root), real);
  if (within.startsWith('..' + path.sep) || within === '..' || path.isAbsolute(within)) {
    throw new Error('That file is outside the learning workspace.');
  }
  return real;
}

function relativeFile(absolute) { return path.relative(root, absolute).split(path.sep).join('/'); }

function bridge(action, payload = {}) {
  const invoke = () => new Promise((resolve, reject) => {
    const python = path.join(root, '.venv', 'Scripts', 'python.exe');
    const child = spawn(python, ['-X', 'utf8', path.join(root, 'tools', 'vscode_bridge.py'), action, '--root', root], {
      cwd: root, windowsHide: true, env: { ...process.env, PYTHONUTF8: '1' }, stdio: ['pipe', 'pipe', 'pipe']
    });
    let out = '', err = '';
    const timer = setTimeout(() => { child.kill(); reject(new Error('The local learning helper took too long.')); }, 60000);
    child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
    child.stdout.on('data', text => { out += text; });
    child.stderr.on('data', text => { err += text; });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', code => {
      clearTimeout(timer);
      try {
        const response = JSON.parse(out);
        if (!response.ok || code !== 0) throw new Error(response.error || 'The local helper failed.');
        resolve(response.result);
      } catch (error) { reject(new Error(out ? error.message : err.trim() || error.message)); }
    });
    child.stdin.end(JSON.stringify(payload));
  });
  if (['complete', 'reading', 'note', 'bookmark', 'create'].includes(action)) {
    const result = writes.then(invoke);
    writes = result.catch(() => {});
    return result;
  }
  return invoke();
}

function pageHtml(webview, config) {
  const media = vscode.Uri.joinPath(context.extensionUri, 'media');
  const script = webview.asWebviewUri(vscode.Uri.joinPath(media, 'app.js'));
  const style = webview.asWebviewUri(vscode.Uri.joinPath(media, 'style.css'));
  const nonce = crypto.randomBytes(18).toString('base64');
  const encoded = JSON.stringify(config).replace(/</g, '\\u003c');
  return `<!doctype html><html lang="en"><head><meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data: https:; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <link rel="stylesheet" href="${style}"><title>Mech Interp Workbench</title></head>
    <body><div id="app"><p class="muted">Opening your learning workspace…</p></div>
    <div id="status" role="status" aria-live="polite"></div>
    <script id="config" type="application/json" nonce="${nonce}">${encoded}</script>
    <script src="${script}" nonce="${nonce}"></script></body></html>`;
}

function bindPanel(panel, config) {
  panel.webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] };
  panel.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'book.svg');
  panel.webview.html = pageHtml(panel.webview, config);
  panel.webview.onDidReceiveMessage(async message => {
    if (!message || !Number.isInteger(message.requestId)) return;
    try {
      const result = await dispatch(message.action, message.payload || {});
      await panel.webview.postMessage({ requestId: message.requestId, result });
    } catch (error) {
      channel.appendLine(`${message.action}: ${error.message}`);
      await panel.webview.postMessage({ requestId: message.requestId, error: error.message });
    }
  }, undefined, context.subscriptions);
}

async function openSource(url) {
  const parsed = new URL(url);
  if (!['https:', 'http:'].includes(parsed.protocol)) throw new Error('Only http and https reading links are supported.');
  const commands = await vscode.commands.getCommands(true);
  if (!commands.includes('workbench.action.browser.open')) {
    throw new Error('This reading needs VS Code’s integrated browser. Update VS Code to use it inside the editor.');
  }
  await vscode.commands.executeCommand('workbench.action.browser.open', { url, openToSide: true });
}

async function openNotebook(file) {
  const relative = relativeFile(file);
  const environment = await bridge('environment', { path: relative });
  const document = await vscode.workspace.openNotebookDocument(vscode.Uri.file(file));
  await vscode.window.showNotebookDocument(document, { viewColumn: vscode.ViewColumn.Beside, preview: false });
  try {
    jupyter ||= await vscode.extensions.getExtension('ms-toolsai.jupyter').activate();
    await jupyter.ready;
    if (typeof jupyter.openNotebook !== 'function') throw new Error('Automatic kernel selection is unavailable in this Jupyter extension version.');
    // The Jupyter API resolves the existing Python path and selects its native notebook controller.
    await jupyter.openNotebook(document.uri, { id: environment.id, path: environment.path });
    channel.appendLine(`Opened ${relative} with ${environment.label}.`);
  } catch (error) {
    channel.appendLine(`Kernel selection: ${error.message}`);
    vscode.window.showWarningMessage(`Choose ${environment.label} in Select Kernel. Interpreter: ${environment.path}`);
  }
  return { path: relative, environment: environment.label };
}

async function openDocument(file) {
  const relative = relativeFile(file), key = 'document:' + relative;
  if (panels.has(key)) { panels.get(key).reveal(vscode.ViewColumn.Beside); return; }
  const panel = vscode.window.createWebviewPanel('mechInterp.document', path.basename(file), vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true });
  panels.set(key, panel);
  bindPanel(panel, { mode: 'document', path: relative });
  panel.onDidDispose(() => panels.delete(key));
}

async function openPaper(id) {
  const course = JSON.parse(fs.readFileSync(path.join(root, 'curriculum.json'), 'utf8'));
  const resource = course.resources.find(r => r.id === id && r.local);
  if (!resource) throw new Error('That paper is not in the local reading library.');
  const key = 'paper:' + id;
  if (panels.has(key)) { panels.get(key).reveal(vscode.ViewColumn.Beside); return; }
  const panel = vscode.window.createWebviewPanel('mechInterp.paper', resource.title, vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true });
  panels.set(key, panel);
  bindPanel(panel, { mode: 'paper', id });
  panel.onDidDispose(() => panels.delete(key));
}

async function openFile(target, base = '') {
  if (!root) throw new Error('Open the Mech Interp workspace folder first.');
  if (!target) {
    const state = await bridge('snapshot');
    const options = [...state.course.modules.map(module => ({ label: module.title, file: module.file })),
      ...state.arena.filter(record => record.kind === 'exercise').map(record => ({ label: record.title, file: record.local }))];
    const selected = await vscode.window.showQuickPick(options, { placeHolder: 'Choose a learning notebook' });
    if (!selected) return;
    target = selected.file;
  }
  if (/^https?:\/\//i.test(target)) return openSource(target);
  if (/^[a-z][a-z\d+.-]*:/i.test(target)) throw new Error('Unsupported learning link.');
  const clean = decodeURIComponent(target.split('#')[0].split('?')[0]);
  const file = workspaceFile(clean, base), relative = relativeFile(file);
  if (relative === '00_Start_Here.ipynb') return openDashboard();
  const course = JSON.parse(fs.readFileSync(path.join(root, 'curriculum.json'), 'utf8'));
  const paper = course.resources.find(r => r.local === relative || r.reader === relative);
  if (paper) return openPaper(paper.id);
  if (file.endsWith('.ipynb')) return openNotebook(file);
  if (file.endsWith('.md')) return openDocument(file);
  await vscode.commands.executeCommand('vscode.open', vscode.Uri.file(file), vscode.ViewColumn.Beside);
}

async function openDashboard(tab = 'continue') {
  if (!root) { vscode.window.showInformationMessage('Open the Mech Interp folder to use its learning dashboard.'); return; }
  if (dashboard) {
    dashboard.reveal(vscode.ViewColumn.One);
    dashboard.webview.postMessage({ type: 'navigate', tab });
    return;
  }
  dashboard = vscode.window.createWebviewPanel('mechInterp.dashboard', 'Learning Dashboard', vscode.ViewColumn.One, { enableScripts: true, retainContextWhenHidden: true });
  bindPanel(dashboard, { mode: 'dashboard', tab });
  dashboard.onDidDispose(() => { dashboard = undefined; });
}

async function dispatch(action, payload) {
  if (action === 'open') { await openFile(payload.target, payload.base); return {}; }
  if (action === 'paperOpen') { await openPaper(payload.id); return {}; }
  if (action === 'dashboard') { await openDashboard(payload.tab); return {}; }
  if (action === 'edit') {
    await vscode.window.showTextDocument(vscode.Uri.file(workspaceFile(payload.path)), { viewColumn: vscode.ViewColumn.Beside, preview: false });
    return {};
  }
  const allowed = ['snapshot', 'document', 'paper', 'complete', 'reading', 'note', 'bookmark', 'create'];
  if (!allowed.includes(action)) throw new Error('Unknown dashboard action.');
  const result = await bridge(action, payload);
  if (action === 'create') await openFile(result.path);
  return result;
}

function activate(extensionContext) {
  context = extensionContext;
  root = workspaceRoot();
  channel = vscode.window.createOutputChannel('Learning Workbench');
  context.subscriptions.push(channel);
  vscode.commands.executeCommand('setContext', 'mechInterp.available', Boolean(root));
  context.subscriptions.push(
    vscode.commands.registerCommand('mechInterp.dashboard', openDashboard),
    vscode.commands.registerCommand('mechInterp.plan', () => openDashboard('plan')),
    vscode.commands.registerCommand('mechInterp.openFile', (file, base) => openFile(file, base)),
    vscode.window.registerUriHandler({ handleUri: async uri => {
      try {
        const params = new URLSearchParams(uri.query);
        if (!root || path.relative(path.resolve(root), path.resolve(params.get('root') || '')) !== '') {
          throw new Error('Open this course’s Mech Interp VS Code window to follow its learning links.');
        }
        await openFile(params.get('path') || params.get('url'));
      } catch (error) { vscode.window.showErrorMessage(error.message); }
    } }),
    vscode.window.registerWebviewPanelSerializer('mechInterp.dashboard', { deserializeWebviewPanel: async (panel, state) => {
      if (!root) { panel.dispose(); return; }
      if (dashboard && dashboard !== panel) dashboard.dispose();
      dashboard = panel;
      bindPanel(panel, { mode: 'dashboard', tab: state?.tab || 'continue' });
      panel.onDidDispose(() => { if (dashboard === panel) dashboard = undefined; });
    } })
  );
  if (root) {
    const items = [['Continue', 'continue', 'play'], ['Study plan', 'plan', 'list-ordered'], ['Course checklists', 'course', 'checklist'],
      ['ARENA notebooks', 'arena', 'notebook'], ['Reading library', 'library', 'book'], ['Your work', 'work', 'beaker']];
    const provider = { getChildren: () => items, getTreeItem: item => {
      const node = new vscode.TreeItem(item[0]);
      node.iconPath = new vscode.ThemeIcon(item[2]);
      node.command = { command: 'mechInterp.dashboard', title: item[0], arguments: [item[1]] };
      return node;
    } };
    context.subscriptions.push(vscode.window.registerTreeDataProvider('mechInterp.navigation', provider));
    const status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
    status.text = '$(book) Learning'; status.command = 'mechInterp.dashboard'; status.tooltip = 'Open your learning dashboard'; status.show();
    context.subscriptions.push(status);
    if (vscode.workspace.getConfiguration('mechInterp').get('openOnStartup', true)) {
      const timer = setTimeout(() => { if (!dashboard) openDashboard(); }, 750);
      context.subscriptions.push({ dispose: () => clearTimeout(timer) });
    }
  }
  return { openDashboard, openFile, openPaper, openSource, bridge };
}

module.exports = { activate };
