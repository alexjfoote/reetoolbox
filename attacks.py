from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
import copy
from constraints import Constraints


class Attack(ABC):
    def __init__(self, model, Transform, hyperparameters, transform_hyperparameters, device="cuda:0"):
        self.model = model
        self.Transform = Transform
        self.hyperparameters = hyperparameters
        self.transform_hyperparameters = transform_hyperparameters
        self.device = device
        self.transform = None

    @abstractmethod
    def run_attack(self):
        pass


class PGD(Attack):
    def run_attack(self, inputs, labels=None, target_classes=None):
        epsilon = self.hyperparameters["epsilon"]
        steps = self.hyperparameters["steps"]
        constraint = self.hyperparameters["constraint"]
        C = self.hyperparameters["C"]
        input_range = self.hyperparameters["input_range"]

        if constraint is not None:
            constraint_func = getattr(Constraints, constraint)
            constraints = Constraints

        inputs = inputs.to(self.device)

        self.transform = self.Transform(input_shape=inputs.shape, device=self.device, **self.transform_hyperparameters)

        in_train_mode = self.model.training
        self.model.eval()

        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False

        opt = torch.optim.RMSprop([self.transform.weights], lr=epsilon)

        for i in range(steps):
            opt.zero_grad()

            if labels is None:
                outputs = self.model(inputs)
                labels = torch.argmax(labels, dim=1)

            original_inputs = copy.deepcopy(inputs)
            adv_inputs = self.transform.forward(inputs)
            inputs = original_inputs
            adv_outputs = self.model(adv_inputs)

            if target_classes is None:
                adv_outputs = adv_outputs.gather(1, labels.unsqueeze(1))[:, 0]
                loss = adv_outputs
            else:
                num_classes = list(adv_outputs.shape)[1]
                all_out = torch.sum(adv_outputs, dim=1)
                target_out = adv_outputs.gather(1, target_classes)[:, 0]
                avg_out = (all_out / num_classes) - target_out
                loss = avg_out

            loss.backward(torch.ones_like(loss), retain_graph=False)
            opt.step()

            if constraint is not None:
                with torch.no_grad():
                    perturbation = self.transform.weights - self.transform.base_weights
                    perturbation = constraint_func(constraints, perturbation, C)
                    self.transform.weights += -self.transform.weights + self.transform.base_weights + perturbation

        original_inputs = copy.deepcopy(inputs)
        adv_inputs = self.transform.forward(inputs)
        inputs = original_inputs

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        if in_train_mode:
            self.model.train()

        return inputs, adv_inputs