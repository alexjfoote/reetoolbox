from sklearn.metrics import accuracy_score
import torch


def accuracy(results):
    labels = results["labels"]
    outputs = results["outputs"]
    _, predictions = torch.max(outputs, 1)
    return accuracy_score(labels, predictions)


def adversarial_accuracy(results):
    labels = results["labels"]
    outputs = results["adversarial_outputs"]
    _, predictions = torch.max(outputs, 1)
    return accuracy_score(labels, predictions)


def rmse(a):
    return torch.sqrt(torch.mean(torch.pow(a, 2)))


def input_sensitivity(results):
    outputs = torch.exp(results["outputs"])
    adv_outputs = torch.exp(results["adversarial_outputs"])
    mean_out_diff = torch.mean(torch.abs(outputs - adv_outputs))
    return mean_out_diff.item()


def normalised_input_sensitivity(results):
    in_sens = input_sensitivity(results)

    pert_measures = results["perturbation_measures"]
    avg_pert = torch.mean(pert_measures)

    norm_in_sens = in_sens / avg_pert
    return norm_in_sens.item()


def fooling_ratio(results):
    acc = accuracy(results)
    adv_acc = adversarial_accuracy(results)
    return (acc - adv_acc) / acc


def fooling_rate(results):
    return fooling_ratio(results) * 100


def get_metrics(results):
    acc = accuracy(results)
    robust_acc = adversarial_accuracy(results)
    fool_ratio = fooling_ratio(results)
    print(f"Accuracy: {acc:.3f}, robust accuracy: {robust_acc:.3f}, fooling ratio: {fool_ratio:.3f}")
