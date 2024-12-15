import sys
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time
import torch.utils.model_zoo as model_zoo
import math


num_workers = int(sys.argv[-1])
class Arguments():
    def __init__(self, no_cuda):
        self.batch_size = 256
        self.test_batch_size = 1000
        self.epochs = 30
        self.lr = 0.05
        self.momentum = 0.5
        self.no_cuda = no_cuda
        self.seed = 1
        self.log_interval = 30
        self.save_model = False

# VGG
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)

        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# Adapted version for Split Learning
def make_layers_SL(cfg, batch_norm=False):
    layers1,layers2 = [],[]
    in_channels = 3
    count=0
    for v in cfg:
        if count<=3:
          if v == 'M':
              count+=1
              layers1 += [nn.MaxPool2d(kernel_size=2, stride=2)]
          else:
              conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
              if batch_norm:
                  layers1 += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
              else:
                  layers1 += [conv2d, nn.ReLU(inplace=True)]
              in_channels = v
        else:
          if v == 'M':
              count +=1
              layers2 += [nn.MaxPool2d(kernel_size=2, stride=2)]
          else:
              conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
              if batch_norm:
                  layers2 += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
              else:
                  layers2 += [conv2d, nn.ReLU(inplace=True)]
              in_channels = v


    return nn.Sequential(*layers1),nn.Sequential(*layers2)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'C': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model

def vgg11_bn_SL(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers_SL(cfg['A'], batch_norm=True)
    model1, model2 = layers[0], VGG(layers[1], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model1, model2

# VGG11 version for split learning, returns two models
def vgg11_SL(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers_SL(cfg['A'])
    model1, model2 = layers[0], VGG(layers[1], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model1, model2

def vgg16_SL(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    layers = make_layers_SL(cfg['D'])
    model1, model2 = layers[0], VGG(layers[1], **kwargs)
    #if pretrained:
    #    model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model1, model2


criterion = nn.CrossEntropyLoss()
def train(args, model, device, federated_train_loader, optimizer, epoch, models, optimizers, clients_mem):
    model.train()
    for batch_idx, (data, targs) in enumerate(federated_train_loader):
        i = int(data.location.id.split()[-1])
        mod_c,opt_c = models[i], optimizers[i]



        # 1) erase previous gradients (if they exist) and update parameters
        optimizer.step()
        opt_c.step()
        opt_c.zero_grad()
        optimizer.zero_grad()

        tg_copy = targs.copy()
        target = tg_copy.get()
        data, target = data.to(device), target.to(device)

        # 2) make a prediction until cut layer (client location)
        pred_c = mod_c(data)
        copy = pred_c.copy()


        # 3) get this to the server
        inp = copy.get()


        # 4) make prediction with second part of the model (server location)
        pred = model(inp)
        # 5) calculate how much we missed

        loss = criterion(pred,target)
        loss.backward()

        gradient = inp.grad

        clients_mem[i] += (gradient.element_size()*gradient.nelement()) + (inp.element_size() * inp.nelement())
        print((gradient.element_size()*gradient.nelement()) + (inp.element_size() * inp.nelement()))

        gradient = gradient.send(data.location)
        pred_c.backward(gradient)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(federated_train_loader) * args.batch_size,
                100. * batch_idx / len(federated_train_loader), loss.item()))


def test(args, model, device, test_loader, models):
    model.eval()
    test_loss = 0
    correct = 0
    n = len(models)
    M = []
    for i in range(n):
        M.append(models[0].copy().get())
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = 0
            for i in range(n):
                output += model(M[i](data))
            output = output/n
            test_loss += criterion(output, target).item()
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def experiment(num_workers,no_cuda):

    # Creating num_workers clients
    clients = []
    hook = sy.TorchHook(torch)
    clients_mem = torch.zeros(num_workers)
    for i in range(num_workers):
        clients.append(sy.VirtualWorker(hook, id="c "+str(i)))


    # Initializing arguments, with GPU usage or not
    args = Arguments(no_cuda)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
    # TODO Quickhack. Actually need to fix the problem moving the model to CUDA\n",
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 0, 'pin_memory': False} if use_cuda else {}


    # Federated data loader
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader
      datasets.CIFAR10('../data', train=True, download=True,
                     transform=transform)
      .federate(clients), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
      batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
      datasets.CIFAR10('../data', train=False, transform=transform),
      batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #creating the models for each client
    models, optimizers = [], []

    for i in range(num_workers):
        #print(i)
        models.append(vgg11_SL()[0].to(device))
        models[i] = models[i].send(clients[i])
        optimizers.append(optim.SGD(params=models[i].parameters(),lr=args.lr,momentum=0.9))


    start = time.time()
    model = vgg11_SL()[1].to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9) # TODO momentum is not supported at the moment

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, federated_train_loader, optimizer, epoch, models, optimizers, clients_mem)
        test(args, model, device, test_loader, models)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    end = time.time()
    print(end - start)
    print("Memory exchanged : ", clients_mem)
    return clients_mem


experiment(num_workers,False)
