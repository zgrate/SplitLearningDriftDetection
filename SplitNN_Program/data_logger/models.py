from django.db import models


# Create your models here.

class DataTransferLog(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    data_transfer_len = models.IntegerField()

    target_source_client = models.CharField(max_length=255, null=True, default=None)

    direction_to_server = models.BooleanField(default=False)
    source_method = models.CharField(max_length=255, null=True, default=None)
