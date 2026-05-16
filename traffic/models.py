from django.db import models

"""
IngestionRun tracks each time we ingest data from the CSV files into the database.
This allows us to monitor the performance and resource usage of the ingestion process over time,
and to keep a history of when data was ingested and how much data was processed.
"""
class IngestionRun(models.Model):
    started_at = models.DateTimeField(auto_now_add=True)
    files_processed = models.PositiveIntegerField(default=0)
    rows_written = models.PositiveBigIntegerField(default=0)
    memory_before_mb = models.FloatField(null=True, blank=True)
    memory_after_mb = models.FloatField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self):
        return f"IngestionRun {self.pk} ({self.started_at:%Y-%m-%d %H:%M})"

"""
ForecastRun tracks the results of running a forecasting model on a specific square.
This allows us to compare the performance of different models on the same square, and to track how
model performance changes over time as we make improvements or changes to the models.
"""
class ForecastRun(models.Model):
    square_id = models.PositiveIntegerField()
    model_name = models.CharField(max_length=32)
    mae = models.FloatField()
    mape = models.FloatField()
    rmse = models.FloatField()
    train_seconds = models.FloatField()
    predict_seconds = models.FloatField()
    hyperparams = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["square_id", "model_name"]
        unique_together = [("square_id", "model_name")]

    def __str__(self):
        return f"{self.model_name} @ square {self.square_id}"

"""
ExperimentRun tracks the results of running a specific experiment, which may involve multiple phases and models.
This allows us to keep a detailed history of our experiments, including the parameters used, the performance
"""
class ExperimentRun(models.Model):
    experiment_id = models.CharField(max_length=64, unique=True)
    phase = models.PositiveSmallIntegerField()
    phase_name = models.CharField(max_length=32)
    model_name = models.CharField(max_length=32)
    square_id = models.PositiveIntegerField()
    params = models.JSONField()
    val_mae = models.FloatField()
    val_mape = models.FloatField()
    val_rmse = models.FloatField()
    train_seconds = models.FloatField()
    reasoning = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["phase", "experiment_id"]

    def __str__(self):
        return self.experiment_id
