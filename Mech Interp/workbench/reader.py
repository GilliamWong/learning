"""A small PDF reader that also works in browsers without a native PDF plug-in."""

from html import escape

import ipywidgets as w
from IPython.display import HTML, display
import pymupdf

from .state import ROOT, Progress, curriculum
from .ui import html, file_link, web_link


def paper_reader(resource_id):
    resource = next(r for r in curriculum()["resources"] if r["id"] == resource_id)
    path = ROOT / resource["local"]
    with pymupdf.open(path) as document:
        count = len(document)
    store = Progress()
    key = f"paper-page-{resource_id}"
    try:
        initial = max(1, min(count, int(store.read().get("notes", {}).get(key, 1))))
    except (ValueError, TypeError):
        initial = 1
    page = w.BoundedIntText(value=initial, min=1, max=count, description="Page:", layout=w.Layout(width="155px"))
    previous = w.Button(description="Previous", icon="arrow-left", layout=w.Layout(width="115px"))
    following = w.Button(description="Next", icon="arrow-right", layout=w.Layout(width="95px"))
    mode = w.ToggleButtons(options=["Page", "Text"], value="Page")
    output = w.Output(layout=w.Layout(width="100%"))
    page_image = w.Image(format="png", layout=w.Layout(width="100%", max_width="1100px"))
    counter = w.HTML()

    def render(_=None):
        previous.disabled = page.value == 1
        following.disabled = page.value == count
        counter.value = f"<small>of {count} · your page is saved when you move</small>"
        with pymupdf.open(path) as document:
            current = document[page.value - 1]
            with output:
                output.clear_output(wait=True)
                if mode.value == "Page":
                    pixmap = current.get_pixmap(dpi=160, alpha=False)
                    page_image.value = pixmap.tobytes("png")
                    display(page_image)
                else:
                    display(HTML('<pre style="white-space:pre-wrap;line-height:1.65">' + escape(current.get_text()) + '</pre>'))

    def changed(change):
        store.note(key, str(change["new"]))
        render()
    page.observe(changed, names="value")
    mode.observe(render, names="value")
    previous.on_click(lambda _: setattr(page, "value", max(1, page.value - 1)))
    following.on_click(lambda _: setattr(page, "value", min(count, page.value + 1)))
    render()
    return w.VBox([
        html(f'<p>{file_link("00_Start_Here.ipynb", "← Home")} · {web_link(resource["url"], "Original source")}</p>'
             f'<h2>{escape(resource["title"])}</h2><p>{escape(resource["target"])}</p>'),
        w.HBox([previous, page, counter, following]), mode, output,
    ], layout=w.Layout(width="100%", max_width="1150px"))
