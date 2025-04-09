from torch import nn, Tensor
from torch.nn import Sequential
from torch.optim import SGD, AdamW


#
#
# class ClientServerModel0:
#
#     client = lambda : nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(784, 400),
#         nn.ReLU(),
#         # nn.Linear(600, 600),
#         # nn.ReLU(),
#         # nn.Linear(600, 500),
#         # nn.ReLU(),
#     )
#
#     server = lambda :nn.Sequential(
#             nn.Linear(400, 400),
#             nn.ReLU(),
#             nn.Linear(400, 200),
#             nn.ReLU(),
#             # nn.Linear(1000, 200),
#             # nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             # nn.Linear(100, out_features=128),
#             # nn.ReLU(),
#             nn.Linear(100, out_features=100),
#             nn.ReLU(),
#             nn.Linear(100, out_features=10),
#         )

#https://medium.com/@BrendanArtley/mnist-keras-simple-cnn-99-6-731b624aee7f



class ClientServerModel0:

    batch_size = 0
    optimiser = SGD
    optimiser_parameters = {"lr": 0.001}

    client = lambda : nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
    )

    server = lambda : nn.Sequential(
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
        )


class ClientServerModel1:

    batch_size = 0
    optimiser = SGD
    optimiser_parameters = {"lr": 0.001}

    client = lambda : nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
        nn.ReLU(),
    )

    server = lambda : nn.Sequential(
            nn.Linear(100, out_features=128),
            nn.ReLU(),
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
        )



# class ClientServerModel1:
#
#     client = lambda : nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(784, 400),
#         nn.ReLU(),
#         nn.Linear(400, 400),
#         nn.ReLU(),
#         nn.Linear(400, 300),
#         nn.ReLU(),
#         nn.Linear(300, 200),
#         nn.ReLU(),
#     )
#
#     server = lambda :nn.Sequential(
#
#             nn.Linear(200, 200),
#             nn.ReLU(),
#             nn.Linear(200, 100),
#             nn.ReLU(),
#             nn.Linear(100, out_features=128),
#             nn.ReLU(),
#             nn.Linear(128, out_features=100),
#             nn.ReLU(),
#             nn.Linear(100, out_features=10),
#         )




# class ClientServerModel2:
#
#
#     client = lambda : nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(784, 400),
#         nn.ReLU(),
#         nn.Linear(400, 400),
#         nn.ReLU(),
#         nn.Linear(400, 300),
#         nn.ReLU(),
#         nn.Linear(300, 200),
#         nn.ReLU(),
#         nn.Linear(200, 200),
#         nn.ReLU(),
#     )
#
#     server = lambda :nn.Sequential(
#         nn.Linear(200, 100),
#         nn.ReLU(),
#         nn.Linear(100, out_features=128),
#         nn.ReLU(),
#         nn.Linear(128, out_features=100),
#         nn.ReLU(),
#         nn.Linear(100, out_features=10),
#     )

class ClientServerModel2:

    batch_size = 0
    optimiser = SGD
    optimiser_parameters = {"lr": 0.001}

    client = lambda : nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
        nn.ReLU(),
        nn.Linear(100, out_features=128),
        nn.ReLU(),
    )

    server = lambda :nn.Sequential(
            nn.Linear(128, out_features=100),
            nn.ReLU(),
            nn.Linear(100, out_features=10),
        )



class ClientServerModel3:

    batch_size = 32
    optimiser = SGD
    optimiser_parameters = {"lr": 0.001}

    client = lambda : nn.Sequential(
        # ClientLayer(),
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 100),
        nn.ReLU(),
        nn.Linear(100, out_features=128),
        nn.ReLU(),
        nn.Linear(128, out_features=100),
        nn.ReLU(),
    )

    server = lambda : nn.Sequential(
        nn.Linear(100, out_features=10),
    )



class CNNClientServerModel1:

    batch_size = 2048
    optimiser = AdamW
    optimiser_parameters = {"lr": 0.001}

    client = lambda : Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        # Input is grayscale, hence in_channels=1
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(0.25)
    )

    server = lambda : Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(0.25),
        nn.Flatten(),

        nn.Linear(3136, 512),  # Update 16x16 if input size differs
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.25),

        nn.Linear(512, 1024),  # Input size (512) should match the output from the previous layer
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 10),  # Output size is 10 (e.g., for classification with 10 classes)
        nn.Softmax(dim=1)
    )


class CNNClientServerModel2:

    batch_size = 32
    optimiser = SGD
    optimiser_parameters = {"lr": 0.001}

    client = lambda : Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        # Input is grayscale, hence in_channels=1
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(0.25),
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Dropout(0.25),
        nn.Flatten(),

    )

    server = lambda : Sequential(
        nn.Linear(512, 1024),  # Input size (512) should match the output from the previous layer
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 10),  # Output size is 10 (e.g., for classification with 10 classes)
        nn.Softmax(dim=1)
    )