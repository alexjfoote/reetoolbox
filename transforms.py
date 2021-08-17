import numpy as np
import torch
import torch.nn.functional as fcn
from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable


class Transform(ABC):
    def __init__(self, input_shape, device):
        """
        Abstract Transform class
        :param input_shape: shape of input batch
        :param device: device that is performing the computations, e.g., cpu, cuda:0
        """
        self.input_shape = input_shape
        self.device = device
        self.base_weights = None
        self.weights = None

    @abstractmethod
    def forward(self, x):
        """
        Performs the differentiable transform on x using self.weights
        :return: x
        """
        pass


class PixelTransform(Transform):
    def __init__(self, input_shape, device, input_range=(0, 255), noise_range=(-0.1, 0.1)):
        super().__init__(input_shape, device)
        self.input_range = input_range
        self.base_weights = torch.zeros(input_shape).to(device)
        weights = torch.FloatTensor(*input_shape).uniform_(*noise_range).to(device)
        self.weights = Variable(weights, requires_grad=True)

    def forward(self, x):
        x = torch.clamp(x + self.weights, *self.input_range)
        return x


class StainTransform(Transform):
    def __init__(self, input_shape, device, input_range=(0, 255)):
        super().__init__(input_shape, device)
        self.max_input_value = input_range[1]

        batch_size = self.input_shape[0]
        self.ruiford = np.array([[0.65, 0.70, 0.29],
                                 [0.07, 0.99, 0.11],
                                 [0.27, 0.57, 0.78]], dtype=np.float)
        self.base_weights = np.stack([self.ruiford] * batch_size)
        self.base_weights = torch.tensor(self.base_weights).to(device)

        self.weights = np.stack([self.ruiford] * batch_size)
        self.weights = torch.tensor(self.weights).to(device)
        self.weights = Variable(self.weights, requires_grad=True)

    def forward(self, x):
        x = self.run_unmix(x)
        x = self.run_mix(x)
        return x

    def run_unmix(self, x):
        # operates in NCHW
        x = (x / self.max_input_value).float()
        outputs = []
        for i, X in enumerate(x):
            Z = self._stain_unmix(X.permute(1, 2, 0), i)
            outputs.append(Z.permute(2, 0, 1))
        x = torch.stack(outputs)
        x = (x * self.max_input_value).float()
        return x

    def run_mix(self, x):
        x = (x / self.max_input_value).float()
        outputs = []
        for i, X in enumerate(x):
            Z = self._stain_remix(X.permute(1, 2, 0), i)
            outputs.append(Z.permute(2, 0, 1))
        x = torch.stack(outputs)
        x = (x * self.max_input_value).float()
        return x

    def _stain_unmix(self, X, index):
        # operates in HWC
        X = torch.max(X, 1e-6 * torch.ones_like(X))
        Z = (torch.log(X) / np.log(1e-6)) @ torch.inverse(self.normalise_matrix(self.base_weights[index]))
        return Z

    def _stain_remix(self, Z, index):
        # operates in HWC
        log_adjust = -np.log(1E-6)
        log_rgb = -(Z * log_adjust) @ self.normalise_matrix(self.weights[index])
        rgb = torch.exp(log_rgb)
        rgb = torch.clamp(rgb, 0, 1)
        return rgb

    def normalise_matrix(self, matrix):
        return fcn.normalize(matrix).float()


class MeanTransform(Transform):
    def __init__(self, input_shape, device, input_range=(0, 255), noise_range=(-0.1, 0.1)):
        super().__init__(input_shape, device)
        self.input_range = input_range
        batch_size = self.input_shape[0]
        shape = (batch_size, 1)
        self.base_weights = torch.zeros(shape).to(device)
        weights = torch.FloatTensor(*shape).uniform_(*noise_range).to(device)
        self.weights = Variable(weights, requires_grad=True).to(device)

    def forward(self, x):
        for i in range(self.input_shape[0]):
            x[i] = torch.clamp(x[i] + self.weights[i], *self.input_range)
        return x
