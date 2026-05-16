from django.core.management.base import BaseCommand

from traffic.services.failure_analysis import run_failure_analysis
from traffic.services.paths import ensure_dirs


class Command(BaseCommand):
    help = "Task 3 failure analysis: residual plots and worst time windows."

    def add_arguments(self, parser):
        parser.add_argument(
            "--window-intervals",
            type=int,
            default=36,
            help="Sliding window size in 10-min steps (36 ≈ 6 hours).",
        )

    def handle(self, *args, **options):
        ensure_dirs()
        result = run_failure_analysis(window_intervals=options["window_intervals"])
        self.stdout.write(self.style.SUCCESS("Failure analysis complete."))
        self.stdout.write(f"Output: {result['output_dir']}")
        self.stdout.write(f"Report: {result['markdown']}")
