from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse
from django.urls import reverse


def _output_root() -> Path:
    return Path(settings.OUTPUT_DIR).resolve()


def _safe_output_path(relative: str) -> Path:
    root = _output_root()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root)):
        raise Http404("Invalid path")
    return target


def _collect_figures() -> dict[str, list[dict]]:
    root = _output_root()
    groups: dict[str, list[dict]] = {
        "eda": [],
        "forecast": [],
        "training": [],
        "experiments": [],
        "failure": [],
    }
    if not root.exists():
        return groups

    for sub, key in [
        ("eda", "eda"),
        ("forecast", "forecast"),
        ("forecast/training", "training"),
        ("experiments", "experiments"),
        ("failure_analysis", "failure"),
    ]:
        folder = root / sub
        if not folder.is_dir():
            continue
        for png in sorted(folder.glob("*.png")):
            rel = png.relative_to(root).as_posix()
            groups[key].append(
                {
                    "name": png.name,
                    "url": reverse("serve_output", kwargs={"filepath": rel}),
                }
            )
    return groups


def _render_gallery(figures: dict[str, list[dict]]) -> str:
    labels = [
        ("eda", "Task 2 — Exploratory analysis"),
        ("forecast", "Task 3 — Forecast vs actual"),
        ("training", "Task 3 — Neural net training loss (per epoch)"),
        ("experiments", "Hyperparameter tuning (validation week)"),
        ("failure", "Task 3 — Failure analysis / residuals"),
    ]
    parts = []
    for key, title in labels:
        items = figures[key]
        if not items:
            continue
        imgs = "".join(
            f'<figure style="display:inline-block;margin:8px;text-align:center;">'
            f'<img src="{i["url"]}" alt="{i["name"]}" '
            f'style="max-width:320px;max-height:220px;border:1px solid #ccc;">'
            f'<figcaption style="font-size:11px;">{i["name"]}</figcaption></figure>'
            for i in items
        )
        parts.append(f"<h2>{title}</h2><div>{imgs}</div>")
    return "".join(parts)


def index(request):
    figures = _collect_figures()
    total = sum(len(v) for v in figures.values())
    gallery = _render_gallery(figures)
    guide_url = reverse("project_guide")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Time Series Forecasting</title></head>
<body style="font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;">
<h1>Milan Mobile Traffic — Dashboard</h1>
<p>Project <code>time_series_forecasting</code> · app <code>traffic</code></p>
<p><strong>{total}</strong> figure(s) · <a href="{guide_url}">Project guide</a></p>
<h2>Pipeline</h2>
<ol>
  <li><code>python manage.py ingest_raw</code> — raw text → Parquet</li>
  <li><code>python manage.py build_series</code> — 10-min series per square</li>
  <li><code>python manage.py run_eda</code> — exploratory plots</li>
  <li><code>python manage.py run_experiments</code> — tune on validation week</li>
  <li><code>python manage.py run_forecast</code> — final test-week evaluation</li>
  <li><code>python manage.py run_failure_analysis</code> — worst intervals + residuals</li>
</ol>
{gallery if gallery else "<p><em>No figures yet. Run the pipeline.</em></p>"}
</body></html>"""
    return HttpResponse(html)


def project_guide(request):
    guide_path = Path(settings.BASE_DIR) / "PROJECT_GUIDE.md"
    if not guide_path.exists():
        raise Http404("PROJECT_GUIDE.md not found")
    body = guide_path.read_text(encoding="utf-8")
    escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Project Guide</title></head>
<body style="font-family:system-ui;max-width:900px;margin:2rem auto;">
<p><a href="{reverse('index')}">← Dashboard</a></p>
<pre style="white-space:pre-wrap;line-height:1.5;font-size:14px;">{escaped}</pre>
</body></html>"""
    return HttpResponse(html)


def serve_output(request, filepath: str):
    target = _safe_output_path(filepath)
    if not target.is_file():
        raise Http404("File not found")
    content_types = {".png": "image/png", ".json": "application/json"}
    content_type = content_types.get(target.suffix, "application/octet-stream")
    return FileResponse(open(target, "rb"), content_type=content_type)
