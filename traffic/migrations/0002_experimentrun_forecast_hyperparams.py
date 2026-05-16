from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("traffic", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastrun",
            name="hyperparams",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.CreateModel(
            name="ExperimentRun",
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
                ("experiment_id", models.CharField(max_length=64, unique=True)),
                ("phase", models.PositiveSmallIntegerField()),
                ("phase_name", models.CharField(max_length=32)),
                ("model_name", models.CharField(max_length=32)),
                ("square_id", models.PositiveIntegerField()),
                ("params", models.JSONField()),
                ("val_mae", models.FloatField()),
                ("val_mape", models.FloatField()),
                ("val_rmse", models.FloatField()),
                ("train_seconds", models.FloatField()),
                ("reasoning", models.TextField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["phase", "experiment_id"]},
        ),
    ]
