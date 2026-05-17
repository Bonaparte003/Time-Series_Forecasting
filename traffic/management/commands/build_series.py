from django.core.management.base import BaseCommand

from traffic.services.etl import run_build_series
from traffic.services.paths import ensure_dirs
from traffic.services.verbose import add_verbose_argument, command_log


class Command(BaseCommand):
    help = "Build per-square time series and metadata (top-traffic + fixed squares)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--save-all",
            action="store_true",
            help="Save all 10k squares (large disk use). Default: target squares only.",
        )
        add_verbose_argument(parser)

    def handle(self, *args, **options):
        ensure_dirs()
        metadata = run_build_series(
            save_all=options["save_all"],
            log=command_log(options, self.stdout.write),
        )
        self.stdout.write(self.style.SUCCESS("Series built."))
        self.stdout.write(f"Top-traffic square: {metadata['top_traffic_square_id']}")
        self.stdout.write(f"Target squares: {metadata['target_square_ids']}")
