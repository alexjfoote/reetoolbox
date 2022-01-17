import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def load_resnet(model_path, n_classes=2, hidden_size=512, pretrained=True, device="cuda:0"):
    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Linear(hidden_size, n_classes),
        nn.LogSoftmax(dim=1))

    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()
    return model


def get_dataloader(dataset, n=100, batch_size=8, num_workers=2, shuffle=False):
    if n is None or n > len(dataset):
        n = len(dataset)
    X_y = dataset[:n]
    X = X_y[0]
    y = X_y[1]
    predict_data = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(predict_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def label_patch(patch, count):
    if count[0] >= 5:
        return patch, 1
    elif count[0] == 0:
        return patch, 0
    else:
        return patch, -1


def load_folds(folds):
    X = []
    y = []
    all_counts = []

    for images, counts in folds:
        for i, image in enumerate(images):
            cc = counts[i]
            image, label = label_patch(image, cc)

            if label == -1:
                continue

            im = Image.fromarray(np.uint8(image))
            X.append(np.transpose(np.array(im.resize((224, 224))), (2, 0, 1)))
            y.append(label)
            all_counts.append(cc)

    X = np.array(X)
    y = np.array(y)
    all_counts = np.array(all_counts)

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return X, y, all_counts


def load_pannuke(data_path, return_counts=False):
    F = np.load(data_path, allow_pickle=True)['F']

    Xtr, ytr, tr_counts = load_folds([F[1], F[2]])
    Xts, yts, ts_counts = load_folds([F[0]])

    if return_counts:
        return Xtr, ytr, Xts, yts, tr_counts, ts_counts
    return Xtr, ytr, Xts, yts


def plot_change(param_values, all_scores, xlabel="", ylabel="", title="", x_range=None, y_range=None, y_ticks=None):
    fig = plt.figure()
    plot = sns.lineplot(x=param_values, y=all_scores)
    plot.set_xlabel(xlabel)
    plot.set_ylabel(ylabel)
    plot.set_title(title)
    if x_range is not None:
        plot.set(xlim=x_range)
    if y_range is not None:
        plot.set(ylim=y_range)
    if y_ticks is not None:
        plot.set(yticks=y_ticks)
    return fig


def display_results(model, input, adv_input, classes):
    in_out = model(input.unsqueeze(0))
    adv_out = model(adv_input.unsqueeze(0))

    in_index = torch.argmax(in_out).item()
    adv_index = torch.argmax(adv_out).item()

    in_class = classes[in_index]
    adv_class = classes[adv_index]

    in_conf = round(torch.exp(in_out)[0][in_index].item() * 100, 1)
    adv_conf = round(torch.exp(adv_out)[0][adv_index].item() * 100, 1)

    perturbation = adv_input - input

    fig = plt.figure(figsize=[13, 9])

    input = input.permute(1,2,0).detach().cpu().numpy()
    adv_input = adv_input.permute(1,2,0).detach().cpu().numpy()
    perturbation = perturbation.permute(1,2,0).detach().cpu().numpy()

    plt.subplot(1, 3, 1)
    plt.imshow(input.astype(int))
    plt.title(in_class + ", " + str(in_conf) + "% confidence")
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])

    plt.subplot(1, 3, 2)
    rmse = np.sqrt(np.mean(np.power(perturbation, 2)))
    l_inf = np.max(np.abs(perturbation))

    min_val = np.min(np.abs(perturbation))
    max_val = np.max(np.abs(perturbation))

    if min_val < 1 and min_val != max_val:
        perturbation -= np.min(perturbation)
        scale_factor = 255/np.max(perturbation)
        perturbation *= scale_factor
    else:
        scale_factor = 1.0
        perturbation = np.abs(perturbation)

    plt.imshow(perturbation.astype(int))
    plt.title(f'Perturbation - scale factor: {scale_factor:.3f}')
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    x_lab = f"L2: {rmse:.3f}, L infinity: {l_inf:.3f}"
    plt.xlabel(x_lab)

    plt.subplot(1, 3, 3)
    plt.imshow(adv_input.astype(int))
    plt.title(adv_class + ", " + str(adv_conf) + "% confidence")
    plt.xticks(ticks=[])
    plt.yticks(ticks=[])

    return fig


def plot_hist(num_epochs, hist, title):
  plt.figure()
  plt.plot(range(num_epochs), hist)
  plt.title(title)
  plt.xlim([0, num_epochs])
  plt.ylim([0, 1])


# Differentiable compression utils from https://github.com/mlomnitz/DiffJPEG
# Used under the MIT license
"""
MIT License

Copyright (c) 2021 Michael R Lomnitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def diff_round(x):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))


class Utils:
    def __init__(self):
        c_table = np.empty((8, 8), dtype=np.float32)
        c_table.fill(99)
        c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                    [24, 26, 56, 99], [47, 66, 99, 99]]).T
        self.c_table = nn.Parameter(torch.from_numpy(c_table)).to("cpu")


utils = Utils()


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        #    result = torch.from_numpy(result)
        result.view(image.shape)
        return result


class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output:
        patch(tensor):  batch x h*w/64 x h x w
    """

    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """

    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        # image = image.float() / (self.c_table**2+eps * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp

        return components['y'], components['cb'], components['cr']


class y_dequantize(nn.Module):
    """ Dequantize Y channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """

    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """ Dequantize CbCr channel
    Inputs:
        image(tensor): batch x height x width
        factor(float): compression factor
    Outputs:
        image(tensor): batch x height x width

    """

    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


class idct_8x8(nn.Module):
    """ Inverse discrete Cosine Transformation
    Input:
        dcp(tensor): batch x height x width
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """ Merge pathces into image
    Inputs:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Output:
        image(tensor): batch x height x width
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """ Upsample chroma layers
    Input:
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Ouput:
        image(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """ Converts YCbCr image to RGB JPEG
    Input:
        image(tensor): batch x height x width x 3
    Outpput:
        result(tensor): batch x 3 x height x width
    """

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
            [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
            dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        # result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """ Full JPEG decompression algortihm
    Input:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        image(tensor): batch x 3 x height x width
    """

    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()

        self.height, self.width = height, width

    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height / 2), int(self.width / 2)
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)
            #
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image / 255


