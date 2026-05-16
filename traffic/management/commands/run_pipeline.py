from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Full pipeline: ingest → build_series → eda → experiments → "
        "forecast → failure analysis."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Limit raw files during ingest (testing).",
        )
        parser.add_argument(
            "--skip-experiments",
            action="store_true",
            help="Skip hyperparameter tuning.",
        )
        parser.add_argument(
            "--quick-experiments",
            action="store_true",
            help="Use smaller experiment grid.",
        )
        parser.add_argument(
            "--skip-forecast",
            action="store_true",
            help="Stop after EDA / experiments.",
        )

    def handle(self, *args, **options):
        ingest_kwargs = {}
        if options["max_files"]:
            ingest_kwargs["max_files"] = options["max_files"]

        call_command("ingest_raw", **ingest_kwargs)
        call_command("build_series")
        call_command("run_eda")

        if not options["skip_experiments"]:
            exp_kwargs = {}
            if options["quick_experiments"]:
                exp_kwargs["quick"] = True
            call_command("run_experiments", **exp_kwargs)

        if not options["skip_forecast"]:
            call_command("run_forecast")
            call_command("run_failure_analysis")

        self.stdout.write(self.style.SUCCESS("Pipeline finished."))
