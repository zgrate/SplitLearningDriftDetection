from django.test import TestCase

from data_logger.factories import TrainingLogFactory
from drift_detection.drift_detectors import SimpleAverageDriftDetection, KolmogorovSmirnovDriftDetection


# Create your tests here.
class DriftDetectionTests(TestCase):
    def test_simple_drift_detection(self):
        TrainingLogFactory.create_batch(10, loss=0.1)
        a = TrainingLogFactory(loss=0.2)

        self.assertTrue(SimpleAverageDriftDetection(0.1, 10, "loss").is_drifting())

    def test_kstest_drift_detection(self):
        TrainingLogFactory.create_batch(15, loss=0.1)
        a = TrainingLogFactory(loss=0.2)

        self.assertTrue(KolmogorovSmirnovDriftDetection().is_drifting())
