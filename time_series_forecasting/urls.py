"""time_series_forecasting URL configuration."""

from django.urls import path

from traffic import views

urlpatterns = [
    path("", views.index, name="index"),
    path("guide/", views.project_guide, name="project_guide"),
    path("outputs/<path:filepath>", views.serve_output, name="serve_output"),
]
