import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def load_resnet(model_path, n_classes=2, pretrained=True, device="cuda:0"):
    model = models.resnet18(pretrained=pretrained)

    model.fc = nn.Sequential(
        nn.Linear(512, n_classes),
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
            image, y = label_patch(image, cc)

            if y == -1:
                continue

            im = Image.fromarray(np.uint8(image))
            X.append(np.transpose(np.array(im.resize((224, 224))), (2, 0, 1)))
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

