import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy


class Metric:
    def __init__(self, metric, attack, model, dataset, batch_size=8, num_workers=2):
        self.metric = metric
        self.attack = attack
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def predict(self, adversarial, n=None):
        if n is None or n > len(self.dataset):
            n = len(self.dataset)

        X_y = self.dataset[:n]
        X = X_y[0]
        y = X_y[1]
        predict_data = torch.utils.data.TensorDataset(X, y)

        dataloader = torch.utils.data.DataLoader(predict_data, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers)

        outputs = []
        all_labels = []
        count = 0
        for inputs, labels in dataloader:
            count += 1
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            if adversarial:
                inputs = self.attack.run_attack(self.model, inputs, labels=labels)

            batch_output = self.model(inputs)
            cpu_output = batch_output.cpu().detach()
            outputs.extend(cpu_output)
            all_labels.extend(labels.cpu().detach())
        outputs = torch.stack(outputs)
        all_labels = torch.stack(all_labels)
        return outputs, all_labels

    def compute_metric(self, adversarial=False, n=None):
        outputs, labels = self.predict(adversarial, n)
        return self.metric(outputs, labels)

    def metric_vs_strength(self, param, param_range, step_size, n=100, do_plot=False, plot_title=None, savepath=None):
        start = param_range[0]
        end = param_range[1] + step_size
        param_values = [f for f in np.arange(start, end, step_size)]

        all_scores = []

        printerval = np.round(len(param_values) / 4)

        original_parameters = copy.deepcopy(self.attack.parameters)

        for i, value in enumerate(param_values):
            self.attack.parameters[param] = value

            if i == 0:
                print("Starting hyperparameters:", self.attack.parameters)
            if (i + 1) % printerval == 0:
                print(f"{round((i + 1) * 100 / len(param_values))}% complete...")

            score = self.compute_metric(n)
            all_scores.append(score)

        print("Done.")

        if do_plot:
            fig = plt.figure()
            plot = sns.lineplot(x=param_values, y=all_scores)
            plot.set_xlabel(param.title())
            plot.set_ylabel("Accuracy/%")
            plot.set(xlim=param_range,
                     ylim=(0, 105),
                     yticks=[0, 20, 40, 60, 80, 100])
            if plot_title is not None:
                plot.set_title(plot_title)

            if savepath is not None:
                fig.savefig(savepath)

        self.attack.parameters = original_parameters

        return param_values, all_scores

    def set_attack_parameters(self, parameters):
        self.attack.parameters = parameters
