from django.shortcuts import render
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.csrf import ensure_csrf_cookie
from django.shortcuts import render

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stage8_review_service as review_service
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import knowledge_query_service
import ollama_evidence_adapter

def _job(job_id):
    """
    Resolve a Django job safely through the pipeline service.
    """

    from pipeline.service import get_job

    job = get_job(job_id)

    if job is None:
        raise Http404("Job not found.")

    return job


def stage8_status(request, job_id):
    job = _job(job_id)
    job_dir = Path(job["job_dir"])

    documents = review_service.get_documents(
        job_dir
    )

    result = []

    for document in documents:
        progress = review_service.get_progress(
            job_dir,
            document,
        )

        queue = review_service.get_review_queue(
            job_dir,
            document,
        )

        result.append(
            {
                "document": document,
                "progress": progress,
                "queue_length": len(queue),
            }
        )

    return JsonResponse(
        {
            "stage": 8,
            "job_id": job_id,
            "documents": result,
        },
        json_dumps_params={
            "ensure_ascii": False,
            "indent": 2,
        },
    )


def stage8_review_item(
    request,
    job_id,
    document_name,
    index,
):
    job = _job(job_id)
    job_dir = Path(job["job_dir"])

    item = review_service.get_review_item(
        job_dir,
        document_name,
        index,
    )

    if item is None:
        return JsonResponse(
            {
                "error": "Review item not found."
            },
            status=404,
        )

    return JsonResponse(
        {
            "stage": 8,
            "job_id": job_id,
            "document": document_name,
            "index": index,
            "item": item,
        },
        json_dumps_params={
            "ensure_ascii": False,
            "indent": 2,
        },
    )


def stage8_review_action(
    request,
    job_id,
    document_name,
    index,
):
    if request.method != "POST":
        return JsonResponse(
            {
                "error": "Only POST is allowed."
            },
            status=405,
        )

    job = _job(job_id)
    job_dir = Path(job["job_dir"])

    item = review_service.get_review_item(
        job_dir,
        document_name,
        index,
    )

    if item is None:
        return JsonResponse(
            {
                "error": "Review item not found."
            },
            status=404,
        )

    try:
        payload = json.loads(
            request.body.decode("utf-8")
        )
    except Exception:
        return JsonResponse(
            {
                "error": "Request body must contain valid JSON."
            },
            status=400,
        )

    action = payload.get("action")

    try:
        if action == "confirm_original":

            result = review_service.confirm_original(
                job_dir,
                document_name,
                item,
            )

        elif action == "confirm_source":

            result = review_service.confirm_source(
                job_dir,
                document_name,
                item,
            )

        elif action == "save_correction":

            verified_text = payload.get(
                "verified_text"
            )

            if verified_text is None:
                return JsonResponse(
                    {
                        "error": (
                            "verified_text is required "
                            "for save_correction."
                        )
                    },
                    status=400,
                )

            result = review_service.save_correction(
                job_dir,
                document_name,
                item,
                verified_text,
            )

        elif action == "skip":

            result = review_service.skip_item(
                job_dir,
                document_name,
                item,
            )

        else:
            return JsonResponse(
                {
                    "error": "Unsupported action."
                },
                status=400,
            )

    except Exception as exc:
        return JsonResponse(
            {
                "error": str(exc)
            },
            status=500,
        )

    return JsonResponse(
        {
            "stage": 8,
            "job_id": job_id,
            "document": document_name,
            "index": index,
            "action": action,
            "result": result,
        },
        json_dumps_params={
            "ensure_ascii": False,
            "indent": 2,
        },
    )


@ensure_csrf_cookie
def stage8_review_page(
    request,
    job_id,
    document_name,
    index,
):
    job = _job(job_id)
    job_dir = Path(job["job_dir"])

    item = review_service.get_review_item(
        job_dir,
        document_name,
        index,
    )

    if item is None:
        progress = review_service.get_progress(job_dir, document_name)
        return render(
            request,
            "review/stage8_review.html",
            {
                "job_id": job_id,
                "document_name": document_name,
                "index": index,
                "item": None,
                "progress": progress,
            },
        )

    progress = review_service.get_progress(
        job_dir,
        document_name,
    )

    return render(
        request,
        "review/stage8_review.html",
        {
            "job_id": job_id,
            "document_name": document_name,
            "index": index,
            "item": item,
            "progress": progress,
        },
    )


def stage8_review_image(
    request,
    job_id,
    document_name,
    index,
):
    job = _job(job_id)
    job_dir = Path(job["job_dir"])

    item = review_service.get_review_item(
        job_dir,
        document_name,
        index,
    )

    if item is None:
        raise Http404(
            "Review item not found."
        )

    review_crop = item.get(
        "review_crop"
    )

    if not review_crop:
        raise Http404(
            "No review crop is available "
            "for this item."
        )

    crop_path = Path(review_crop)

    if not crop_path.is_file():
        raise Http404(
            "Review crop file does not exist."
        )

    return FileResponse(
        open(crop_path, "rb"),
        content_type="image/png",
    )

def knowledge_query(request, job_id):
    if request.method != "GET":
        return JsonResponse(
            {
                "error": "Only GET is allowed."
            },
            status=405,
        )

    query = request.GET.get("q", "").strip()

    if not query:
        return JsonResponse(
            {
                "error": "Query parameter 'q' is required."
            },
            status=400,
        )

    try:
        result = knowledge_query_service.query_knowledge(
            job_id,
            query,
        )
    except FileNotFoundError as exc:
        return JsonResponse(
            {
                "error": str(exc)
            },
            status=404,
        )
    except Exception as exc:
        return JsonResponse(
            {
                "error": str(exc)
            },
            status=500,
        )

    return JsonResponse(
        result,
        json_dumps_params={
            "ensure_ascii": False,
            "indent": 2,
        },
    )

def knowledge_explorer(request, job_id):
    return render(
        request,
        "review/knowledge_query.html",
        {
            "job_id": job_id,
        },
    )



def knowledge_explanation(request, job_id):
    if request.method != "GET":
        return JsonResponse(
            {"error": "Only GET is allowed."},
            status=405,
        )

    query = request.GET.get("q", "").strip()
    if not query:
        return JsonResponse(
            {"error": "Query parameter 'q' is required."},
            status=400,
        )

    try:
        retrieval = knowledge_query_service.query_knowledge(
            job_id,
            query,
        )
        result = ollama_evidence_adapter.explain_evidence(
            query,
            retrieval,
        )
        return JsonResponse(
            result,
            json_dumps_params={
                "ensure_ascii": False,
                "indent": 2,
            },
        )
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
