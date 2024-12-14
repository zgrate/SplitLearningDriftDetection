import math

from data_logger.models import TrainingLog

class DetectorBase:
    def is_drifting(self) -> bool:
        raise NotImplemented("Please implement")

    def get_last_object(self):
        return TrainingLog.objects.order_by('-created_at').first()

    def get_last_X_values(self, count: int):
        return TrainingLog.objects.exclude(id=self.get_last_object().id).order_by('-created_at')[:self.count]

class SimpleAverageDriftDetection(DetectorBase):

    def __init__(self, threshold=0.1, last_elements_average=10, average_field="loss"):
        super(SimpleAverageDriftDetection, self).__init__()
        self.last_elements_average = last_elements_average
        self.average_field = average_field
        self.threshold = threshold

    def is_drifting(self) -> bool:
        print(TrainingLog.objects.count())
        last = TrainingLog.objects.order_by('-created_at').first()
        last_10 = TrainingLog.objects.exclude(id=last.id).order_by('-created_at')[
                  :self.last_elements_average].values_list(self.average_field, flat=True)
        value = last.__dict__[self.average_field]
        average = sum(last_10) / len(last_10)
        print(math.fabs(average - value))
        return math.fabs(average - value) >= self.threshold
