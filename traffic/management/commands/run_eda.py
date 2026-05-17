from django.core.management.base import BaseCommand

from traffic.services.eda import run_eda
from traffic.services.paths import ensure_dirs
from traffic.services.verbose import add_verbose_argument, command_log


class Command(BaseCommand):
    help = "Task 2: generate exploratory analysis figures."

    def add_arguments(self, parser):
        add_verbose_argument(parser)

    def handle(self, *args, **options):
        from django.conf import settings

        ensure_dirs()
        outputs = run_eda(log=command_log(options, self.stdout.write))
        self.stdout.write(self.style.SUCCESS(f"Generated {len(outputs)} EDA artifacts."))
        self.stdout.write(f"Output directory: {settings.OUTPUT_DIR / 'eda'}")
