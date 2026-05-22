from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from traffic.db_insights import serialize_experiments
from traffic.presentation import build_presentation_context


def _output_root() -> Path:
    return Path(settings.OUTPUT_DIR).resolve()


def _safe_output_path(relative: str) -> Path:
    root = _output_root()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root)):
        raise Http404("Invalid path")
    return target


def _safe_doc_image_path(relative: str) -> Path:
    root = (Path(settings.BASE_DIR) / "docs" / "images").resolve()
    target = (root / relative).resolve()
    if not str(target).startswith(str(root)):
        raise Http404("Invalid path")
    return target


def serve_output(request, filepath: str):
    target = _safe_output_path(filepath)
    if not target.is_file():
        raise Http404("File not found")
    content_types = {".png": "image/png", ".json": "application/json"}
    content_type = content_types.get(target.suffix, "application/octet-stream")
    return FileResponse(open(target, "rb"), content_type=content_type)


def serve_doc_image(request, filepath: str):
    target = _safe_doc_image_path(filepath)
    if not target.is_file():
        raise Http404("File not found")
    return FileResponse(open(target, "rb"), content_type="image/png")


def presentation(request):
    return render(
        request,
        "traffic/presentation.html",
        build_presentation_context(),
    )


@require_GET
def api_experiments(request):
    """Filter experiment runs from SQLite (used by presentation JS)."""
    model = request.GET.get("model") or None
    phase_raw = request.GET.get("phase")
    phase = int(phase_raw) if phase_raw and phase_raw.isdigit() else None
    limit = min(int(request.GET.get("limit", 40)), 100)
    return JsonResponse(
        {
            "rows": serialize_experiments(model=model, phase=phase, limit=limit),
            "filters": {"model": model, "phase": phase},
        }
    )
