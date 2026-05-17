from django.core.management import call_command
from django.core.management.base import BaseCommand

from traffic.services.verbose import add_verbose_argument, is_verbose


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
        add_verbose_argument(parser)
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

    def _subcommand_kwargs(self, options: dict) -> dict:
        kwargs = {}
        if is_verbose(options):
            kwargs["verbose"] = True
        if options.get("verbosity", 1) >= 2:
            kwargs["verbosity"] = options["verbosity"]
        return kwargs

    def handle(self, *args, **options):
        sub = self._subcommand_kwargs(options)

        ingest_kwargs = dict(sub)
        if options["max_files"]:
            ingest_kwargs["max_files"] = options["max_files"]

        if is_verbose(options):
            self.stdout.write("=== ingest_raw ===")
        call_command("ingest_raw", **ingest_kwargs)

        if is_verbose(options):
            self.stdout.write("\n=== build_series ===")
        call_command("build_series", **sub)

        if is_verbose(options):
            self.stdout.write("\n=== run_eda ===")
        call_command("run_eda", **sub)

        if not options["skip_experiments"]:
            exp_kwargs = dict(sub)
            if options["quick_experiments"]:
                exp_kwargs["quick"] = True
            if is_verbose(options):
                self.stdout.write("\n=== run_experiments ===")
            call_command("run_experiments", **exp_kwargs)

        if not options["skip_forecast"]:
            if is_verbose(options):
                self.stdout.write("\n=== run_forecast ===")
            call_command("run_forecast", **sub)
            if is_verbose(options):
                self.stdout.write("\n=== run_failure_analysis ===")
            call_command("run_failure_analysis", **sub)

        self.stdout.write(self.style.SUCCESS("Pipeline finished."))
