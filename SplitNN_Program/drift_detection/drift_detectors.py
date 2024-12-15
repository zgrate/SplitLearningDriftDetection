import math

import scipy.stats

from data_logger.models import TrainingLog


class DetectorBase:

    def __init__(self, threshold=0.1, count_elements=10, key_comparator="loss"):
        self.count_elements = count_elements
        self.key_comparator = key_comparator
        self.threshold = threshold

    def is_drifting(self) -> bool:
        raise NotImplemented("Please implement")

    def get_last_object(self):
        return TrainingLog.objects.order_by('-created_at').first()

    def get_last_object_value(self):
        return self.get_last_object().__dict__[self.key_comparator]

    def get_last_X_objects_filer_last(self, count_elements=None):
        return self.get_objects_without_last()[:(self.count_elements if count_elements is None else count_elements)]

    def get_objects_without_last(self):
        return self.get_objects().exclude(id=self.get_last_object().id)

    def get_objects(self):
        return TrainingLog.objects.order_by('-created_at')

    def get_objects_values(self):
        return self.get_objects().values_list(self.key_comparator, flat=True)

    def get_last_X_values_filter_last(self, count_elements=None, key: str = "loss"):
        return self.get_last_X_objects_filer_last(count_elements).values_list(self.key_comparator if key is None else key,
                                                                   flat=True)


class SimpleAverageDriftDetection(DetectorBase):

    # def __init__(self, threshold=0.1, last_elements_average=10, average_field="loss"):
    #     super(SimpleAverageDriftDetection, self).__init__(threshold, last_elements_average, average_field)

    def is_drifting(self) -> bool:
        print(TrainingLog.objects.count())
        last = self.get_last_object_value()
        last_10 = self.get_last_X_values_filter_last()

        average = sum(last_10) / len(last_10)

        return math.fabs(average - last) >= self.threshold


class KolmogorovSmirnovDriftDetection(DetectorBase):

    # def __init__(self, threshold=0.1, count_elements=10, key_field="loss"):
    #     super().__init__(count_elements, key_field)
    #     self.threshold = threshold

    def is_drifting(self) -> bool:
        last = self.get_last_object_value()
        objs = self.get_objects()
        last_recent_values = objs.values_list(self.key_comparator, flat=True)[0:self.count_elements]
        last_all_values = objs.values_list(self.key_comparator, flat=True)[self.count_elements:]

        stat = (scipy.stats.kstest(last_recent_values, last_all_values))
        print(stat)
        return False

