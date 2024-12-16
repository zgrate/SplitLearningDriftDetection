import dataclasses

from django.db import models


# Create your models here.

class DataTransferLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    data_transfer_len = models.IntegerField()

    target_source_client = models.CharField(max_length=255, null=True, default=None)

    direction_to_server = models.BooleanField(default=False)
    source_method = models.CharField(max_length=255, null=True, default=None)


class TrainingLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    client_id = models.CharField(max_length=255, null=True, default=None)

    mode = models.CharField(max_length=255, null=True, default='training',
                            choices=[(x, x) for x in ['training', 'reset', 'validation', 'error', 'test']])

    loss = models.FloatField(default=0)
    epoch = models.IntegerField(default=0)
    server_epoch = models.IntegerField(default=0)

    training_time = models.FloatField(default=0)
    last_communication_time = models.FloatField(default=0)
    last_whole_training_time = models.FloatField(default=0)


class DriftingLogger(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    client_id = models.CharField(max_length=255, null=True, default=None)
    client_forced = models.BooleanField(default=False)

    server_epoch = models.IntegerField(default=0)
    client_epoch = models.IntegerField(default=0)

    last_loss = models.FloatField(default=0)
    detection_mode = models.CharField(max_length=255, null=True, default=None)
    detection_parameters = models.JSONField(default=dict, blank=True, null=True)

    calculated_value = models.FloatField(default=0)
    old_value = models.FloatField(default=0)

    comparing_value = models.FloatField(default=0)

    is_drifting = models.BooleanField(default=False)

    correction_policy = models.CharField(max_length=255, null=True, default=None)
