import torch

class Norms:
    def l0(self, perturbation, proportion):
        gradients = abs(perturbation.grad)
        flat = torch.flatten(gradients)
        k = int(proportion * len(flat))
        top_k = torch.topk(flat, k)
        if len(top_k[0]) == 0:
            threshold = 256
        else:
            threshold = top_k[0][-1]
        perturbation[gradients < threshold] = 0
        return perturbation

    def l1(self, perturbation, max_mae):
        mae = torch.mean(torch.abs(perturbation))
        if mae > max_mae:
            multiplier = max_mae / mae
            perturbation *= multiplier
        return perturbation

    def l2(self, perturbation, max_rmse):
        rmse = torch.sqrt(torch.mean(torch.pow(torch.round(perturbation), 2)))
        if rmse > max_rmse:
            multiplier = max_rmse / rmse
            perturbation *= multiplier
        return perturbation

    def l_inf(self, perturbation, max_diff):
        perturbation[:] = torch.clamp(perturbation, -max_diff, max_diff)
        return perturbation
