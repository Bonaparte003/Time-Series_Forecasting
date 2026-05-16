"""
WSGI config for time_series_forecasting project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "time_series_forecasting.settings")

application = get_wsgi_application()
