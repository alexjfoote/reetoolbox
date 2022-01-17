from abc import ABC, abstractmethod
import torch
from reetoolbox.constraints import Constraints


def untargeted_loss(outputs, labels):
    loss = outputs.gather(1, labels.unsqueeze(1))[:, 0]
    return loss


def targeted_loss(outputs, targets):
    num_classes = list(outputs.shape)[1]
    all_out = torch.sum(outputs, dim=1)
    target_out = outputs.gather(1, targets)[:, 0]
    loss = (all_out / num_classes) - target_out
    # loss = -target_out
    return loss


class Optimiser(ABC):
    def __init__(self, model, Transform, hyperparameters, transform_hyperparameters, criterion=untargeted_loss,
                 device="cuda:0"):
        self.model = model
        self.Transform = Transform
        self.hyperparameters = hyperparameters
        self.transform_hyperparameters = transform_hyperparameters
        self.device = device
        self.transform = None
        self.criterion = criterion

    @abstractmethod
    def optimise(self):
        pass


class PGD(Optimiser):
    def optimise(self, inputs, targets=None, reset_weights=True):
        epsilon = self.hyperparameters["epsilon"]
        steps = self.hyperparameters["steps"]
        constraint = self.hyperparameters["constraint"]
        C = self.hyperparameters["C"]
        input_range = self.hyperparameters["input_range"]

        if constraint is not None:
            constraint_func = getattr(Constraints, constraint)
            constraints = Constraints

        inputs = inputs.to(self.device)

        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)

        in_train_mode = self.model.training
        self.model.eval()

        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False

        opt = torch.optim.RMSprop([self.transform.weights], lr=epsilon)

        for i in range(steps):
            opt.zero_grad()

            if targets is None:
                outputs = self.model(inputs)
                targets = torch.argmax(outputs, dim=1)

            original_inputs = inputs.clone()
            adv_inputs = self.transform.forward(inputs)
            inputs = original_inputs
            adv_outputs = self.model(adv_inputs)

            loss = self.criterion(adv_outputs, targets)

            loss.backward(torch.ones_like(loss), retain_graph=True)
            opt.step()

            if constraint is not None:
                self.transform.weights = constraint_func(constraints, self.transform.weights,
                                                         self.transform.base_weights, C)

        original_inputs = inputs.clone()
        adv_inputs = self.transform.forward(inputs)
        inputs = original_inputs

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        if in_train_mode:
            self.model.train()

        return inputs, adv_inputs


class StochasticSearch(Optimiser):
    def optimise(self, inputs, targets=None, reset_weights=True):
        samples = self.hyperparameters["samples"]
        weight_ranges = self.hyperparameters["weight_ranges"]
        input_range = self.hyperparameters["input_range"]

        inputs = inputs.to(self.device)

        if self.transform is None or reset_weights:
            self.transform = self.Transform(input_shape=inputs.shape, device=self.device,
                                            **self.transform_hyperparameters)
            self.best_loss = None
            self.best_adv_inputs = inputs

        in_train_mode = self.model.training
        self.model.eval()

        grads = []
        for param in self.model.parameters():
            grads.append(param.requires_grad)
            param.requires_grad = False

        with torch.no_grad():
            for i in range(samples):
                for j, weight_name in enumerate(weight_ranges):
                    self.transform.weights[weight_name] = torch.FloatTensor(
                        *self.transform.weights[weight_name].shape).uniform_(*weight_ranges[weight_name]).to(self.device)

                if targets is None:
                    outputs = self.model(inputs)
                    targets = torch.argmax(outputs, dim=1)

                original_inputs = inputs.clone()
                adv_inputs = self.transform.forward(inputs)
                inputs = original_inputs
                adv_outputs = self.model(adv_inputs)

                loss = self.criterion(adv_outputs, targets)

                if self.best_loss is None or self.best_adv_inputs is None:
                    self.best_loss = loss.clone()
                    self.best_adv_inputs = adv_inputs.clone()
                else:
                    for j, input_loss in enumerate(loss):
                        if input_loss < self.best_loss[j]:
                            self.best_adv_inputs[j] = adv_inputs[j]
                            self.best_loss[j] = input_loss

        original_inputs = inputs.clone()
        inputs = original_inputs

        for i, param in enumerate(self.model.parameters()):
            param.requires_grad = grads[i]

        if in_train_mode:
            self.model.train()

        return inputs, self.best_adv_inputs
