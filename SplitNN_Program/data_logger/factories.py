import random

import factory.django

from data_logger.models import TrainingLog


class TrainingLogFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = TrainingLog

    client_id = factory.Faker("pyint", min_value=0, max_value=10000)

    mode = "test"

    loss = factory.Faker("pyfloat", min_value=0, max_value=10)

    epoch = factory.Sequence(lambda n: n)
    server_epoch = factory.Sequence(lambda n: n*3)

    training_time = factory.Faker("pyint", min_value=0, max_value=10)
    last_communication_time = 1
    last_whole_training_time = 1