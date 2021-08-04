from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
from norms import Norms

class Attack(ABC):
    def __init__(self, parameters):
        self.parameters = parameters

    @abstractmethod
    def run_attack(self):
        pass


class PGD(Attack):
    def run_attack(self, model, inputs, labels=None, target_classes=None):
        epsilon = self.parameters["epsilon"]
        steps = self.parameters["steps"]
        norm = self.parameters["norm"]
        C = self.parameters["C"]
        margin = self.parameters["margin"]
        input_range = self.parameters["input_range"]

        if norm is not None:
            norm_func = getattr(Norms, norm)
            norms = Norms

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = inputs.to(device)

        for param in model.parameters():
            param.requires_grad = False

        shape = inputs.shape
        noise_range = 0.5
        perturbation = torch.FloatTensor(shape).uniform_(-noise_range, noise_range).to(device)
        perturbation = Variable(perturbation, requires_grad=True)

        opt = torch.optim.RMSprop([perturbation], lr=epsilon)

        count = 0

        for i in range(steps):
            count += 1
            opt.zero_grad()
            adv_inputs = torch.clamp(inputs + perturbation, *input_range)

            if labels is None:
                outputs = model(inputs)
                labels = torch.argmax(labels, dim=1)

            adv_outputs = model(adv_inputs)

            if target_classes is None:
                adv_outputs = adv_outputs.gather(1, labels.unsqueeze(1))[:, 0]
                loss = torch.max(margin - (1 - adv_outputs), torch.zeros_like(labels))
            else:
                num_classes = list(adv_outputs.shape)[1]
                all_out = torch.sum(adv_outputs, dim=1)
                target_out = adv_outputs.gather(1, target_classes)[:, 0]
                avg_out = (all_out / num_classes) - target_out

            loss.backward(torch.ones_like(loss), retain_graph=True)
            opt.step()

            if norm is not None:
                with torch.no_grad():
                    perturbation = norm_func(norms, perturbation, C)

        for param in model.parameters():
            param.requires_grad = True

        adv_inputs = torch.clamp(inputs + perturbation, *input_range)

        return adv_inputs
