import torch.nn.functional as fcn
from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
import torchvision.transforms.functional as TF
from reetoolbox.utils import *


class Transform(ABC):
    def __init__(self, input_shape, device):
        """
        Abstract Transform class
        :param input_shape: shape of input batch
        :param device: device that is performing the computations, e.g., cpu, cuda:0
        """
        self.input_shape = input_shape
        self.device = device
        self.weights = None

    @abstractmethod
    def forward(self, x):
        """
        Performs the transform on x using self.weights
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
    def __init__(self, input_shape, device, input_range=(0, 255), noise_range=(0, 0)):
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
        self.weights += torch.FloatTensor(*self.weights.shape).uniform_(*noise_range).to(device)
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


class CutoutTransform(Transform):
    def __init__(self, input_shape, device, height, width):
        super().__init__(input_shape, device)
        self.height = height
        self.width = width
        batch_size = input_shape[0]
        self.weights = {
            "corner_x": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "corner_y": torch.stack([torch.tensor([0.0])] * batch_size).to(device)
        }

    def forward(self, x):
        for i, _ in enumerate(x):
            corner_y = self.weights["corner_y"][i, 0].int()
            corner_x = self.weights["corner_x"][i, 0].int()
            x[i, :, corner_y:corner_y + self.height, corner_x:corner_x + self.width] = 0
        return x


class RotateTransform(Transform):
    def __init__(self, input_shape, device):
        super().__init__(input_shape, device)
        batch_size = input_shape[0]
        self.weights = {
            "angle": torch.stack([torch.tensor([0.0])] * batch_size).to(device)
        }

    def forward(self, x):
        for i, _ in enumerate(x):
            x[i] = TF.rotate(x[i], self.weights["angle"][i].item())
        return x


class CropTransform(Transform):
    def __init__(self, input_shape, device):
        super().__init__(input_shape, device)
        batch_size = input_shape[0]
        self.weights = {
            "top": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "left": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "height": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "width": torch.stack([torch.tensor([0.0])] * batch_size).to(device)
        }

    def forward(self, x):
        for i, _ in enumerate(x):
            top = int(self.weights["top"][i].item())
            left = int(self.weights["left"][i].item())
            height = int(self.weights["height"][i].item())
            width = int(self.weights["width"][i].item())
            cropped = TF.crop(x[i], top=top, left=left, height=height, width=width)

            left_pad = round((self.input_shape[2] - width) / 2)
            right_pad = self.input_shape[2] - width - left_pad
            top_pad = round((self.input_shape[3] - height) / 2)
            bottom_pad = self.input_shape[3] - height - top_pad
            x[i] = TF.pad(cropped, (left_pad, top_pad, right_pad, bottom_pad))
        return x


class BlurTransform(Transform):
    def __init__(self, input_shape, device):
        super().__init__(input_shape, device)
        batch_size = input_shape[0]
        self.weights = {
            "corner_x": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "corner_y": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "height": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "width": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "kernel_size": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
            "sigma": torch.stack([torch.tensor([0.0])] * batch_size).to(device)
        }

    def forward(self, x):
        for i, _ in enumerate(x):
            corner_y = self.weights["corner_y"][i, 0].int()
            corner_x = self.weights["corner_x"][i, 0].int()
            height = self.weights["height"][i, 0].int()
            width = self.weights["width"][i, 0].int()
            kernel_size = int(self.weights["kernel_size"][i, 0].item())
            if kernel_size % 2 == 0:
                kernel_size += 1
            sigma = self.weights["sigma"][i, 0].item()
            blur_section = TF.gaussian_blur(x[i, :, corner_y:corner_y + height, corner_x:corner_x + width], kernel_size,
                                            sigma)
            x[i, :, corner_y:corner_y + height, corner_x:corner_x + width] = blur_section
        return x


class ZoomInTransform(Transform):
    def __init__(self, input_shape, device):
        super().__init__(input_shape, device)
        batch_size = input_shape[0]
        self.weights = {
            "scale": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
        }

    def forward(self, x):
        for i, _ in enumerate(x):
            scale = self.weights["scale"][i].item()
            big = TF.resize(x[i], (int(self.input_shape[2] * scale), int(self.input_shape[3] * scale)))

            top = int(torch.FloatTensor(1).uniform_(0, big.shape[1] - self.input_shape[2]).to(device).item())
            left = int(torch.FloatTensor(1).uniform_(0, big.shape[2] - self.input_shape[3]).to(device).item())

            x[i] = TF.crop(big, top=top, left=left, height=self.input_shape[2], width=self.input_shape[3])
        return x


class ZoomOutTransform(Transform):
    def __init__(self, input_shape, device):
        super().__init__(input_shape, device)
        batch_size = input_shape[0]
        self.weights = {
            "scale": torch.stack([torch.tensor([0.0])] * batch_size).to(device),
        }

    def forward(self, x):
        for i, _ in enumerate(x):
            scale = self.weights["scale"][i].item()
            small = TF.resize(x[i], (int(self.input_shape[2] * scale), int(self.input_shape[3] * scale)))

            width = small.shape[1]
            height = small.shape[2]
            left_pad = round((self.input_shape[2] - width) / 2)
            right_pad = self.input_shape[2] - width - left_pad
            top_pad = round((self.input_shape[3] - height) / 2)
            bottom_pad = self.input_shape[3] - height - top_pad
            x[i] = TF.pad(small, (left_pad, top_pad, right_pad, bottom_pad))
        return x


class HEDTransform(Transform):
    def __init__(self, input_shape, device, input_range=(0, 255)):
        super().__init__(input_shape, device)
        batch_size = self.input_shape[0]
        self.input_range = input_range
        self.weights = {
            "alpha": torch.zeros(batch_size, 3).to(device),
            "beta": torch.zeros(batch_size, 3).to(device)
        }

        self.max_input_value = input_range[1]
        self.ruifrock = torch.tensor(np.array([[0.65, 0.70, 0.29],
                                 [0.07, 0.99, 0.11],
                                 [0.27, 0.57, 0.78]], dtype=np.float)).to(device)

    def forward(self, x):
        x = (x / self.max_input_value).float()
        x = self.run_unmix(x)
        for i, hed in enumerate(x):
            for j, channel in enumerate(hed):
                hed[j] = (self.weights["alpha"][i, j].cpu() * channel) + self.weights["beta"][i, j].cpu()
            x[i] = hed
        x = self.run_mix(x)
        x = (x * self.max_input_value).float()
        return x

    def run_unmix(self, x):
        # operates in NCHW
        outputs = []
        for i, X in enumerate(x):
            Z = self._stain_unmix(X.permute(1, 2, 0), i)
            outputs.append(Z.permute(2, 0, 1))
        x = torch.stack(outputs)
        return x

    def run_mix(self, x):
        outputs = []
        for i, X in enumerate(x):
            Z = self._stain_remix(X.permute(1, 2, 0), i)
            outputs.append(Z.permute(2, 0, 1))
        x = torch.stack(outputs)
        return x

    def _stain_unmix(self, X, index):
        # operates in HWC
        X = torch.max(X, 1e-6 * torch.ones_like(X))
        Z = (torch.log(X) / np.log(1e-6)) @ torch.inverse(self.normalise_matrix(self.ruifrock))
        return Z

    def _stain_remix(self, Z, index):
        # operates in HWC
        log_adjust = -np.log(1E-6)
        log_rgb = -(Z * log_adjust) @ self.normalise_matrix(self.ruifrock)
        rgb = torch.exp(log_rgb)
        rgb = torch.clamp(rgb, 0, 1)
        return rgb

    def normalise_matrix(self, matrix):
        return fcn.normalize(matrix).float()


class RandomStainTransform(Transform):
    def __init__(self, input_shape, device, input_range=(0, 255)):
        super().__init__(input_shape, device)
        self.max_input_value = input_range[1]

        batch_size = self.input_shape[0]
        self.ruiford = np.array([[0.65, 0.70, 0.29],
                                 [0.07, 0.99, 0.11],
                                 [0.27, 0.57, 0.78]], dtype=np.float)
        self.base_weights = np.stack([self.ruiford] * batch_size)
        self.base_weights = torch.tensor(self.base_weights).to(device)

        self.weights = {
            "weights": torch.stack([torch.zeros((3, 3))] * batch_size).to(device),
        }

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
        log_rgb = -(Z * log_adjust) @ self.normalise_matrix(self.base_weights[index] + self.weights["weights"][index])
        rgb = torch.exp(log_rgb)
        rgb = torch.clamp(rgb, 0, 1)
        return rgb

    def normalise_matrix(self, matrix):
        return fcn.normalize(matrix).float()


class JPEGTransform(Transform):
    def __init__(self, input_shape, device):
        super().__init__(input_shape, device)
        self.rounding = diff_round
        batch_size = self.input_shape[0]
        self.weights = {
            "quality": torch.stack([torch.tensor([100.0])] * batch_size).to("cpu")
        }

    def forward(self, x):
        '''
        '''
        x = x.to("cpu")
        for key in self.weights:
            self.weights[key] = self.weights[key].to("cpu")

        for i, _ in enumerate(x):
            self.factor = quality_to_factor(self.weights["quality"][i].item())
            self.compress = compress_jpeg(rounding=self.rounding, factor=self.factor)
            self.decompress = decompress_jpeg(self.input_shape[2], self.input_shape[3], rounding=self.rounding,
                                              factor=self.factor)
            x_i = x[i].unsqueeze(0) / 255
            y, cb, cr = self.compress(x_i)
            x_i = self.decompress(y, cb, cr)
            x[i] = x_i[0] * 255

        x = x.to(self.device)
        for key in self.weights:
            self.weights[key] = self.weights[key].to(self.device)
        return x
