import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy


class Evaluator:
    def __init__(self, model, dataset, dataloader, TransformOptimiser, Transform, attack_params, trans_params, device="cuda:0"):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = device
        self.trans_params = trans_params
        self.attack = TransformOptimiser(self.model, Transform, attack_params, trans_params, device=device)
        self.model.eval()

    def predict(self, adversarial, perturbation_measure=None, weight_measure=None):
        outputs = []
        adv_outputs = []
        all_labels = []
        pert_measures = []
        weight_measures = []

        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if adversarial:
                inputs, adv_inputs = self.attack.run_attack(inputs, labels=labels)
                batch_output = self.model(adv_inputs).cpu().detach()
                adv_outputs.extend(batch_output)

                if perturbation_measure is not None:
                    input_pert = adv_inputs - inputs
                    pert_measures.append(perturbation_measure(input_pert))
                if weight_measure is not None:
                    weight_pert = self.attack.transform.weights - self.attack.transform.base_weights
                    weight_measures.append(weight_measure(weight_pert))

            batch_output = self.model(inputs).cpu().detach()
            outputs.extend(batch_output)
            all_labels.extend(labels.cpu().detach())

        return_dict = {}
        return_dict["outputs"] = torch.stack(outputs)
        return_dict["labels"] = torch.stack(all_labels)
        if adversarial:
          return_dict["adversarial_outputs"] = torch.stack(adv_outputs)
        if pert_measures:
            return_dict["perturbation_measures"] = torch.stack(pert_measures)
        if weight_measures:
            return_dict["weight_measures"] = torch.stack(weight_measures)
        return return_dict

    def compute_metric(self, metric, **parameters):
        results = self.predict(**parameters)
        score = metric(results)
        return score

    def metric_vs_strength(self, param, param_range, step_size, metric, **metric_params):
        start = param_range[0]
        end = param_range[1] + step_size
        param_values = [f for f in np.arange(start, end, step_size)]

        all_scores = []

        printerval = np.round(len(param_values) / 4)

        original_parameters = copy.deepcopy(self.attack.hyperparameters)

        for i, value in enumerate(param_values):
            self.attack.hyperparameters[param] = value

            if i == 0:
                print("Starting hyperparameters:", self.attack.hyperparameters)
            if i % printerval == 0:
                print(f"{round(i * 100 / len(param_values))}% complete...")

            score = self.compute_metric(metric, **metric_params)
            all_scores.append(score)

        print("Done.")

        self.attack.hyperparameters = original_parameters

        return param_values, all_scores

    def attack_inputs(self, input_indices):
        inputs = []
        labels = []

        for i in input_indices:
            inputs.append(self.dataset[i][0])
            labels.append(self.dataset[i][1])

        inputs = torch.stack(inputs).to(self.device)
        labels = torch.stack(labels).to(self.device)

        inputs, adv_inputs = self.attack.run_attack(inputs, labels=labels)
        return inputs, adv_inputs

    def set_attack_hyperparameters(self, hyperparameters):
        self.attack.hyperparameters = hyperparameters
