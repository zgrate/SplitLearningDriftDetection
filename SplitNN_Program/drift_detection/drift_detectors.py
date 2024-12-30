import math
import statistics

import scipy.stats
import sklearn
import numpy as np

from data_logger.models import TrainingLog, DriftingLogger


class DetectorBase:

    def __init__(self, threshold=0.1, count_elements=10, key_comparator="loss", filter_mode=None, client_only_mode=False):
        self.count_elements = count_elements
        self.key_comparator = key_comparator
        self.threshold = threshold
        self.filter_mode = filter_mode
        self.client_only_mode = client_only_mode

    def is_drifting(self) -> bool:
        raise NotImplemented("Please implement drifting detector")

    def get_last_object(self) -> TrainingLog:
        return self._get_objects().first()

    def get_last_object_value(self):
        return self.get_last_object().__dict__[self.key_comparator]

    def get_last_X_objects_filer_last(self, count_elements=None):
        return self.get_objects_without_last()[:(self.count_elements if count_elements is None else count_elements)]

    def get_last_X_objects(self, count_elements=None):
        return self.get_objects()[:(self.count_elements if count_elements is None else count_elements)]

    def get_objects_without_last(self):
        return self.get_objects().exclude(id=self.get_last_object().id)

    def _get_objects(self):
        t = TrainingLog.objects.order_by('-created_at')
        if self.filter_mode is not None:
            t = t.filter(mode=self.filter_mode)

        return t

    def get_objects(self):
        t = self._get_objects()
        if self.client_only_mode and (first := self.get_last_object()) is not None:
            t = t.filter(client_id=first.client_id)

        return t

    def get_objects_values(self):
        return self.get_objects().values_list(self.key_comparator, flat=True)

    def get_last_X_values(self, count_elements=None, key: str = "loss"):
        return self.get_last_X_objects(count_elements).values_list(self.key_comparator if key is None else key,
                                                                   flat=True)
    def get_last_X_values_filter_last(self, count_elements=None, key: str = "loss"):
        return self.get_last_X_objects_filer_last(count_elements).values_list(self.key_comparator if key is None else key,
                                                                   flat=True)

    def report_drifting(self, old_value, new_value, drift_value, is_drifting):
        last = self.get_last_object()

        p = DriftingLogger(client_id=last.client_id, server_epoch=last.server_epoch, client_epoch=last.epoch, last_loss=last.loss, detection_mode=type(self).__name__, detection_parameters=self.__dict__, old_value=old_value, calculated_value= new_value, comparing_value=drift_value, is_drifting=is_drifting, correction_policy=None)
        p.save()

        print(p.__dict__)


    def is_enough_data(self):
        print(self.get_last_object(), self.get_objects().count())
        return self.get_last_object() is not None and self.get_objects().count() >= self.count_elements

    def get_regression(self):
        if self.is_enough_data():
            a = self.get_last_X_values()
            print(a)
            fit = np.polyfit(range(len(a)), a, deg=1)
            print(fit[0], fit[0] > 0.1)
            return fit[0], fit[0] > 0.1

class LastRoundCompareDriftDetector(DetectorBase):
    def is_drifting(self) -> bool:
        values = self.get_last_X_values(count_elements=2)

        drift_value =  values[0] - values[1]

        is_drifting = drift_value > self.threshold
        self.report_drifting(values[1], values[0], drift_value, is_drifting)
        return is_drifting

class SimpleAverageDriftDetection(DetectorBase):

    # def __init__(self, threshold=0.1, last_elements_average=10, average_field="loss"):
    #     super(SimpleAverageDriftDetection, self).__init__(threshold, last_elements_average, average_field)

    def is_drifting(self) -> bool:
        last = self.get_last_object_value()
        last_10 = self.get_last_X_values_filter_last()

        average = sum(last_10) / len(last_10)
        is_drifting = (drift_value := math.fabs(average - last)) >= self.threshold

        self.report_drifting(average, last, drift_value, is_drifting)

        return is_drifting

class SimpleVarianceDriftDetection(DetectorBase):

    def is_drifting(self) -> bool:
        last_with = self.get_objects_values()
        last_without = self.get_last_X_values_filter_last()

        variance = statistics.variance(last_with)
        variance_2 = statistics.variance(last_without)
        is_drifting = (drift_value := math.fabs(variance - variance_2)) >= self.threshold

        self.report_drifting(variance_2, variance, drift_value, is_drifting)

        return is_drifting


class KolmogorovSmirnovDriftDetection(DetectorBase):

    # def __init__(self, threshold=0.1, count_elements=10, key_field="loss"):
    #     super().__init__(count_elements, key_field)
    #     self.threshold = threshold

    #TODO REPORTING
    def is_drifting(self) -> bool:
        raise NotImplemented("Please implement reporting")
        last = self.get_last_object_value()
        objs = self.get_objects()
        last_recent_values = objs.values_list(self.key_comparator, flat=True)[0:self.count_elements]
        last_all_values = objs.values_list(self.key_comparator, flat=True)[self.count_elements:]

        stat = (scipy.stats.kstest(last_recent_values, last_all_values))
        print(stat)
        return False





class DriftDetectionSuite:

    def __init__(self, drift_detection_class):
        self.drift_detection_class = drift_detection_class
        self.drift_correction_policy = None

    def drift_detection_run(self):
        if self.drift_detection_class.is_enough_data():
            return self.drift_detection_class.is_drifting()

        return False


SimpleAverageDriftDetection().get_regression()