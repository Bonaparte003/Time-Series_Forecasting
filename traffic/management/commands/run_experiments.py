from django.core.management.base import BaseCommand

from traffic.services.experiments import run_experiments
from traffic.services.paths import ensure_dirs
from traffic.services.verbose import add_quiet_argument, add_verbose_argument, command_log


class Command(BaseCommand):
    help = (
        "Hyperparameter tuning: Phase 1 baseline + Phase 2 grid search on "
        "validation week (Dec 9–15). Writes experiments_log.csv and best_hyperparams.json."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--square-id",
            type=int,
            default=None,
            help="Square used for tuning (default: top-traffic square).",
        )
        parser.add_argument(
            "--models",
            nargs="+",
            default=None,
            help="Models to tune: arima lstm tcn",
        )
        add_verbose_argument(parser)
        add_quiet_argument(parser)
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Use smaller hyperparameter grid (faster smoke test).",
        )

    def handle(self, *args, **options):
        ensure_dirs()
        models = tuple(options["models"]) if options["models"] else None
        result = run_experiments(
            square_id=options["square_id"],
            models=models,
            log=command_log(options, self.stdout.write, default=True),
            quick=options["quick"],
        )
        self.stdout.write(self.style.SUCCESS("Experiments complete."))
        self.stdout.write(f"Tuning square: {result['tuning_square_id']}")
        self.stdout.write(f"Runs: {result['n_experiments']}")
        self.stdout.write(f"Log: {result['experiments_csv']}")
        self.stdout.write(f"Best params: {result['best_params_path']}")
        self.stdout.write(f"Journal: {result['journal_path']}")
        self.stdout.write(
            "\nNext: python manage.py run_forecast  "
            "(uses best_hyperparams.json by default)"
        )
