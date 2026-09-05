"""A small notebook UI; curriculum content lives in curriculum.json and notebooks."""

from datetime import datetime
from html import escape
import json
from urllib.parse import quote, urlparse

import ipywidgets as w
from IPython.display import HTML, display

from .state import Progress, curriculum, new_notebook

STYLE = """<style>
.wb {max-width:1000px;font-family:system-ui,sans-serif;color:var(--jp-ui-font-color1,#202a30)}
.wb h1 {font-size:30px;letter-spacing:-.7px;margin:8px 0 12px;line-height:1.2}
.wb h2 {font-size:22px;letter-spacing:-.3px;margin:8px 0 12px}
.wb p {font-size:14px;line-height:1.6;margin:8px 0}
.wb .eyebrow {font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#418476;font-weight:700}
.wb .hero {background:var(--jp-layout-color1,#f7f9f7);border:1px solid #9dbab2;border-left:5px solid #418476;border-radius:8px;padding:22px;margin:12px 0}
.wb .card {border:1px solid var(--jp-border-color2,#ddd);border-radius:7px;padding:16px;margin:10px 0}
.wb .muted {color:var(--jp-ui-font-color2,#53646a);font-size:12px}
.wb a {color:var(--jp-content-link-color,#176e65);font-weight:600}
.wb a.cta,.wb a.cta:visited {display:inline-block;padding:10px 15px;border-radius:5px;color:#fff !important;background:#286e60;text-decoration:none;margin-top:8px}
.wb .pill {font-size:11px;padding:3px 8px;border:1px solid #9dbab2;border-radius:20px;margin-left:6px;white-space:nowrap}
.wb ul {line-height:1.8;padding-left:20px}
</style>"""


def html(body):
    # Jupyter's rich-output renderer supplies proper local-file link handling.
    output = w.Output()
    with output:
        display(HTML(f'{STYLE}<div class="wb">{body}</div>'))
    return output


def file_link(path, label, css=""):
    args = escape(json.dumps({"path": path}), quote=True)
    return (f'<a class="{css}" href="/lab/tree/{quote(path, safe="/")}" '
            f'data-commandlinker-command="docmanager:open" data-commandlinker-args="{args}">{escape(label)}</a>')


def web_link(url, label):
    return f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{escape(label)} ↗</a>'


def note_box(store, key, label, placeholder=""):
    box = w.Textarea(value=store.read().get("notes", {}).get(key, ""), placeholder=placeholder,
                     layout=w.Layout(width="100%", height="95px"))
    status = w.HTML('<small>Notes save here automatically. Save notebook code separately with Ctrl+S.</small>')
    def save(change):
        try:
            store.note(key, change["new"])
            status.value = f'<small>Saved locally at {datetime.now():%H:%M:%S}.</small>'
        except Exception as error:
            status.value = f'<strong>Not saved: {escape(str(error))}</strong>'
    box.observe(save, names="value")
    return w.VBox([html(f'<p><b>{escape(label)}</b></p>'), box, status])


def resource_card(resource, store):
    links = web_link(resource["url"], "Original source")
    if resource.get("local"):
        links = file_link(resource.get("reader", resource["local"]), "Read local paper") + " &nbsp;·&nbsp; " + links
    if resource.get("web"):
        links += " &nbsp;·&nbsp; " + web_link(resource["web"], "Web edition")
    text = html(f'<div class="card"><div class="eyebrow">{escape(resource["type"])} · {escape(resource["author"])}</div>'
                f'<h2 style="font-size:18px">{escape(resource["title"])}</h2><p>{escape(resource["target"])}</p>'
                f'<p class="muted">{escape(resource["depth"])} · {escape(resource["note"])}</p><p>{links}</p></div>')
    status = w.Dropdown(options=["Not started", "Reading", "Assigned sections reviewed", "Revisit"],
                        value=store.read().get("reading", {}).get(resource["id"], "Not started"), description="Reading:")
    status.observe(lambda change: store.reading(resource["id"], change["new"]), names="value")
    return w.VBox([text, status])


def lesson_panel(module_id):
    store = Progress()
    course = curriculum()
    module = next(m for m in course["modules"] if m["id"] == module_id)
    done = store.read().get("completed", {}).get(module_id, [])
    boxes = []
    status = w.HTML('<small>Tick a task when you can explain what you did. Saved automatically.</small>')
    for task in module["tasks"]:
        box = w.Checkbox(value=task["id"] in done, description=task["label"], indent=False,
                         layout=w.Layout(width="100%"), style={"description_width": "initial"})
        def changed(change, task_id=task["id"]):
            try:
                store.complete(module_id, task_id, change["new"])
                status.value = '<small>Progress saved. Use Refresh on the home page to update its next action.</small>'
            except Exception as error:
                status.value = f'<strong>Not saved: {escape(str(error))}</strong>'
        box.observe(changed, names="value")
        boxes.append(box)
    readings = [resource_card(r, store) for r in course["resources"] if r["module"] == module_id]
    accordion = w.Accordion(children=[w.VBox(boxes + [status]), w.VBox(readings)])
    accordion.set_title(0, "My progress")
    accordion.set_title(1, "Readings and official exercises")
    accordion.selected_index = None
    return w.VBox([html(f'<p>{file_link("00_Start_Here.ipynb", "← Home")} <span class="pill">{escape(module["kind"])}</span></p>'),
                   accordion, note_box(store, f"module-{module_id}", "Next time, start here", "Leave a specific next step, or a question you want to return to.")],
                  layout=w.Layout(max_width="1000px"))


def home():
    store = Progress()
    course = curriculum()
    summary = w.VBox()
    route = w.VBox()
    def refresh(_=None):
        state = store.read()
        count = total = 0
        next_item = None
        cards = []
        for module in course["modules"]:
            done = set(state.get("completed", {}).get(module["id"], []))
            complete = sum(task["id"] in done for task in module["tasks"])
            count += complete
            total += len(module["tasks"])
            if next_item is None:
                next_task = next((t for t in module["tasks"] if t["id"] not in done), None)
                if next_task:
                    next_item = module, next_task
            cards.append(html(f'<div class="card"><div class="eyebrow">Module {module["id"]} · {complete}/{len(module["tasks"])} tasks</div>'
                              f'<h2>{file_link(module["file"], module["title"])}</h2><p>{escape(module["subtitle"])}</p>'
                              f'<p class="muted">{escape(module["kind"])} · {escape(module["suggestion"])}</p></div>'))
        if next_item:
            module, task = next_item
            body = f'<div class="hero"><div class="eyebrow">Continue · Module {module["id"]}</div><h1>{escape(module["title"])}</h1>'
            body += f'<p><b>Your next action:</b> {escape(task["label"])}.</p>'
            checkpoint = state.get("notes", {}).get(f'module-{module["id"]}', "")
            if checkpoint:
                body += f'<p><b>Your stopping point:</b> {escape(checkpoint)}</p>'
            body += file_link(module["file"], "Continue learning →", "cta") + '</div>'
        else:
            body = '<div class="hero"><h1>Your next question</h1><p>The current route is complete. Start a small experiment in Your work.</p></div>'
        summary.children = [html(body + f'<p class="muted">{count} of {total} tasks checked · progress is saved in this folder</p>')]
        route.children = [html('<p>Modules 1–2 are complete local exercise sets. Modules 3–6 provide reading plans, official exercise links, and research scaffolds. Follow the route or open any module.</p>')] + cards
    refresh()
    refresh_button = w.Button(description="Refresh progress", icon="refresh")
    refresh_button.on_click(refresh)
    continue_tab = w.VBox([summary, refresh_button, note_box(store, "session", "A note for your next session", "What did you learn? Where would you like to pick up?")])

    library_body = w.VBox()
    select = w.Dropdown(options=[("All resources", "all")] + [(f'{m["id"]} · {m["title"]}', m["id"]) for m in course["modules"]] + [("Later reading", "later")],
                        value="01", description="Show:", layout=w.Layout(width="90%"))
    def show_library(_=None):
        library_body.children = [resource_card(r, store) for r in course["resources"] if select.value in {"all", r["module"]}]
        if not library_body.children:
            library_body.children = [html('<p>Use the module guide to choose sources for your own question.</p>')]
    select.observe(show_library, names="value")
    show_library()
    library_tab = w.VBox([html('<p>Read only the assigned sections for the task you are doing. Local PDFs open beside your notebooks; web articles open at their original source.</p>'), select, library_body])

    created = w.VBox()
    def show_created():
        rows = [f'<li>{file_link(p, p)}</li>' for p in reversed(store.read().get("created", []))]
        created.children = [html('<ul>' + ''.join(rows) + '</ul>' if rows else '<p class="muted">New experiment and paper notebooks will appear here.</p>')]
    make_experiment = w.Button(description="New experiment", icon="flask", button_style="success")
    make_paper = w.Button(description="New paper note", icon="file-text-o")
    def create(template, folder):
        new_notebook(template, folder)
        show_created()
    make_experiment.on_click(lambda _: create("Experiment.ipynb", "experiments"))
    make_paper.on_click(lambda _: create("Paper_Notes.ipynb", "notes"))
    show_created()
    bookmark_title = w.Text(placeholder="Title", layout=w.Layout(width="100%"))
    bookmark_url = w.Text(placeholder="https://…", layout=w.Layout(width="100%"))
    bookmark_reason = w.Text(placeholder="Why save it? When might it help?", layout=w.Layout(width="100%"))
    bookmark_button = w.Button(description="Save resource", icon="bookmark")
    bookmarks = w.HTML()
    bookmark_status = w.HTML()
    def show_bookmarks():
        bookmarks.value = '<ul>' + ''.join(f'<li>{web_link(b["url"], b["title"])} — {escape(b["reason"])}</li>' for b in store.read().get("bookmarks", [])) + '</ul>'
    def save_bookmark(_):
        url = bookmark_url.value.strip()
        parsed = urlparse(url)
        if not bookmark_title.value.strip() or parsed.scheme not in {"http", "https"} or not parsed.netloc:
            bookmark_status.value = '<small>Add a title and a complete http or https address.</small>'
            return
        record = {"title": bookmark_title.value.strip(), "url": url, "reason": bookmark_reason.value.strip()}
        store.update(lambda data: data.setdefault("bookmarks", []).append(record))
        bookmark_title.value = bookmark_url.value = bookmark_reason.value = ""
        bookmark_status.value = '<small>Resource saved.</small>'
        show_bookmarks()
    bookmark_button.on_click(save_bookmark)
    show_bookmarks()
    work_tab = w.VBox([html('<h2>Your work</h2><p>Create a fresh notebook from a small template. The template stays unchanged.</p>'),
                        w.HBox([make_experiment, make_paper]), created,
                        note_box(store, "questions", "Questions to return to", "Capture an idea without interrupting your current task."),
                        html('<h2>Save any other resource</h2>'), bookmark_title, bookmark_url, bookmark_reason, bookmark_button, bookmark_status, bookmarks])
    tabs = w.Tab(children=[continue_tab, route, library_tab, work_tab])
    for index, title in enumerate(["Continue", "Course", "Reading library", "Your work"]):
        tabs.set_title(index, title)
    return w.VBox([html('<div class="eyebrow">Your course & research notebook</div><h1>Mech Interp Workbench</h1><p>Build a model. Read with a question. Test an explanation.</p>'), tabs],
                  layout=w.Layout(width="100%", max_width="1000px"))
