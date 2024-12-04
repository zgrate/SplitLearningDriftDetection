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

    mode = models.CharField(max_length=255, null=True, default='training', choices=[(x,x) for x in ['training', 'reset', 'validation', 'error']])

    loss = models.FloatField(default=0)
    epoch = models.IntegerField(default=0)
    server_epoch = models.IntegerField(default=0)

    training_time = models.FloatField(default=0)
    last_communication_time = models.FloatField(default=0)
    last_whole_training_time = models.FloatField(default=0)
