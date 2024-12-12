import math

from data_logger.models import TrainingLog
from drift_detection.detection.detector_base import DetectorBase


class SimpleAverageDriftDetection(DetectorBase):

    def __init__(self, threshold=0.1, last_elements_average=10, average_field="loss"):
        super(SimpleAverageDriftDetection, self).__init__()
        self.last_elements_average = last_elements_average
        self.average_field = average_field
        self.threshold = threshold

    def is_drifting(self) -> bool:
        last = TrainingLog.objects.order_by('-created_at').first().values_list(self.average_field, flat=True)[0]
        last_10 = TrainingLog.objects.exclude(id=last.id).order_by('-created_at')[:self.last_elements_average].values_list(self.average_field, flat=True)
        average = sum(last_10) / len(last_10)

        return math.fabs(average-last) >= self.threshold
