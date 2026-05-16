import json

from django.core.management.base import BaseCommand

from traffic.models import IngestionRun
from traffic.services.loader import ingest_files
from traffic.services.paths import ensure_dirs, raw_data_files


class Command(BaseCommand):
    help = "Task 1: ingest raw TIM text files into daily Parquet shards."

    def add_arguments(self, parser):
        parser.add_argument(
            "--max-files",
            type=int,
            default=None,
            help="Process only the first N files (for quick tests).",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help=(
                "Print per-file and per-chunk progress while ingesting. "
                "Also enabled when Django verbosity is 2+ (e.g. manage.py ingest_raw -v 2)."
            ),
        )

    def handle(self, *args, **options):
        from django.conf import settings

        ensure_dirs()
        files = raw_data_files()
        daily_dir = settings.PROCESSED_DIR / "daily"

        self.stdout.write(f"Found {len(files)} raw files.")
        verbose = options["verbose"] or options.get("verbosity", 1) >= 2
        log = self.stdout.write if verbose else None
        stats = ingest_files(
            files,
            daily_dir,
            max_files=options["max_files"],
            log=log,
        )

        run = IngestionRun.objects.create(
            files_processed=stats["files_processed"],
            rows_written=stats["rows_written"],
            memory_before_mb=stats["memory_before_mb"],
            memory_after_mb=stats["memory_after_mb"],
            notes=json.dumps(stats, indent=2),
        )

        report_path = settings.PROCESSED_DIR / "ingest_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        self.stdout.write(self.style.SUCCESS(f"Ingestion complete (run id={run.pk})."))
        self.stdout.write(
            f"Memory sample: {stats['memory_before_mb']:.2f} MB → "
            f"{stats['memory_after_mb']:.2f} MB"
        )
        self.stdout.write(f"Rows written: {stats['rows_written']:,}")
        self.stdout.write(f"Daily shards: {daily_dir}")
        self.stdout.write(f"Report: {report_path}")
