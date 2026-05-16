from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="IngestionRun",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("started_at", models.DateTimeField(auto_now_add=True)),
                ("files_processed", models.PositiveIntegerField(default=0)),
                ("rows_written", models.PositiveBigIntegerField(default=0)),
                ("memory_before_mb", models.FloatField(blank=True, null=True)),
                ("memory_after_mb", models.FloatField(blank=True, null=True)),
                ("notes", models.TextField(blank=True)),
            ],
            options={"ordering": ["-started_at"]},
        ),
        migrations.CreateModel(
            name="ForecastRun",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("square_id", models.PositiveIntegerField()),
                ("model_name", models.CharField(max_length=32)),
                ("mae", models.FloatField()),
                ("mape", models.FloatField()),
                ("rmse", models.FloatField()),
                ("train_seconds", models.FloatField()),
                ("predict_seconds", models.FloatField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["square_id", "model_name"],
                "unique_together": {("square_id", "model_name")},
            },
        ),
    ]
