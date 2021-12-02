import torch

class Constraints:
    def l0(self, weights, base_weights, proportion):
        perturbation = weights - base_weights
        gradients = abs(weights.grad)
        flat = torch.flatten(gradients)
        k = int(proportion * len(flat))
        top_k = torch.topk(flat, k)
        if len(top_k[0]) == 0:
            threshold = 256
        else:
            threshold = top_k[0][-1]
        with torch.no_grad():
            perturbation[gradients < threshold] = 0
            weights += -weights + base_weights + perturbation
        return weights

    def l1(self, weights, base_weights, max_mae):
        perturbation = weights - base_weights
        with torch.no_grad():
            mae = torch.mean(torch.abs(perturbation))
            if mae > max_mae:
                multiplier = max_mae / mae
                perturbation *= multiplier
            weights += -weights + base_weights + perturbation
        return weights

    def l2(self, weights, base_weights, max_rmse):
        perturbation = weights - base_weights
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean(torch.pow(perturbation, 2)))
            if rmse > max_rmse:
                multiplier = max_rmse / rmse
                perturbation *= multiplier
            weights += -weights + base_weights + perturbation
        return weights

    def l_inf(self, weights, base_weights, max_diff):
        perturbation = weights - base_weights
        with torch.no_grad():
            perturbation[:] = torch.clamp(perturbation, -max_diff, max_diff)
            weights += -weights + base_weights + perturbation
        return weights

    def in_range(self, weights, base_weights, range):
        return torch.clamp(weights, *range)
