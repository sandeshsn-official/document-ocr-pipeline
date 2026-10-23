from django.urls import path
from . import views


urlpatterns = [
    path(
        "<str:job_id>/stage8/status/",
        views.stage8_status,
        name="stage8_status",
    ),

    path(
        "<str:job_id>/stage8/<str:document_name>/<int:index>/",
        views.stage8_review_item,
        name="stage8_review_item",
    ),

    path(
        "<str:job_id>/stage8/<str:document_name>/<int:index>/action/",
        views.stage8_review_action,
        name="stage8_review_action",
    ),

    path(
        "<str:job_id>/stage8/<str:document_name>/<int:index>/review/",
        views.stage8_review_page,
        name="stage8_review_page",
    ),

    path(
        "<str:job_id>/stage8/<str:document_name>/<int:index>/image/",
        views.stage8_review_image,
        name="stage8_review_image",
    ),

    path(
        "<str:job_id>/knowledge/query/",
        views.knowledge_query,
        name="knowledge_query",
    ),

    path(
        "<str:job_id>/knowledge/explain/",
        views.knowledge_explanation,
        name="knowledge_explanation",
    ),


    path(
        "<str:job_id>/knowledge/",
        views.knowledge_explorer,
        name="knowledge_explorer",
    ),

]