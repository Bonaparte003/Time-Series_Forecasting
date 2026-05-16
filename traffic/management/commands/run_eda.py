from django.core.management.base import BaseCommand

from traffic.services.eda import run_eda
from traffic.services.paths import ensure_dirs


class Command(BaseCommand):
    help = "Task 2: generate exploratory analysis figures."

    def handle(self, *args, **options):
        from django.conf import settings

        ensure_dirs()
        outputs = run_eda()
        self.stdout.write(self.style.SUCCESS(f"Generated {len(outputs)} EDA artifacts."))
        self.stdout.write(f"Output directory: {settings.OUTPUT_DIR / 'eda'}")
