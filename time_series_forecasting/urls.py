"""time_series_forecasting URL configuration."""

from django.urls import path

from traffic import views

urlpatterns = [
    path("", views.presentation, name="presentation"),
    path("api/csv/", views.api_csv_data, name="api_csv_data"),
    path("api/experiments/", views.api_experiments, name="api_experiments"),
    path("outputs/<path:filepath>", views.serve_output, name="serve_output"),
    path("docs-images/<path:filepath>", views.serve_doc_image, name="serve_doc_image"),
]