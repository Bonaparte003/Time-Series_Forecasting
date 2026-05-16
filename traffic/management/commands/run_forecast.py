from django.core.management.base import BaseCommand

from traffic.services.forecast_runner import run_all_forecasts
from traffic.services.paths import ensure_dirs


class Command(BaseCommand):
    help = "Task 3: train and evaluate ARIMA, LSTM, and TCN on target squares."

    def add_arguments(self, parser):
        parser.add_argument(
            "--models",
            nargs="+",
            default=None,
            help="Subset of models: arima lstm tcn",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress per-epoch terminal output.",
        )
        parser.add_argument(
            "--no-best-params",
            action="store_true",
            help="Ignore best_hyperparams.json; use defaults from settings.",
        )

    def handle(self, *args, **options):
        ensure_dirs()
        models = tuple(options["models"]) if options["models"] else None
        result = run_all_forecasts(
            models=models,
            verbose=not options["quiet"],
            use_best_params=not options["no_best_params"],
        )
        self.stdout.write(self.style.SUCCESS("Forecasting complete."))
        self.stdout.write(f"Squares: {result['square_ids']}")
        self.stdout.write(f"Models: {result['models']}")
        self.stdout.write(f"Outputs: {result['output_dir']}")
        self.stdout.write(f"Timing table: {result['timing_table']}")
        self.stdout.write(f"Hardware info: {result['hardware']}")
        self.stdout.write("\nRun: python manage.py run_failure_analysis")
