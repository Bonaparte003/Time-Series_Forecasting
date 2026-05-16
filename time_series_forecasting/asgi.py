"""
ASGI config for time_series_forecasting project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "time_series_forecasting.settings")

application = get_asgi_application()
