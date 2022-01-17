# Robustness Evaluation and Enhancement Toolbox

The Robustness Evaluation and Enhancement Toolbox (REEToolbox) provides tools for measuring and improving the robustness of ML models. REEToolbox uses adversarial optimisation with data transforms to generate challenging transformations of input data. This allows you to measure robustness by evaluating the degradation in performance on the transformed data compared to the input data, or improve robustness by using adversarially optimised transforms for data augmentation during training.



## Usage
There are in-depth tutorial notebooks that demonstrate how to use the toolbox for robustness evaluation and adversarial training in the Tutorials directory.

The basic process is:

Install the toolbox:

`pip install reetoolbox`

Perform a robustness evaluation:

```
# import a transform, an optimiser, the evaluator class, a metric function, and the default evaluation parameters
from reetoolbox.transforms import StainTransform
from reetoolbox.optimisers import PGD
from reetoolbox.evaluator import Evaluator
from reetoolbox.metrics import fooling_rate
from reetoolbox.constants import eval_stain_transform_params, eval_stain_optimiser_params

import torch

# Load your data
Xts, yts = load_data(...)

# Load your model
model = load_model(...)

# Create a PyTorch dataset and dataloader
test_dataset = torch.utils.data.TensorDataset(Xts, yts)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=2)

# Create an evaluator object
stain_evaluator = Evaluator(model, test_dataset, test_loader, PGD, 
                            StainTransform, eval_stain_optimiser_params, 
                            eval_stain_transform_params)

# Use the evaluator to get the model's predictions on the original and transformed data
results = stain_evaluator.predict(adversarial=True)

# Compute the desired metric to get a measure of robustness to the given transform
fr = fooling_rate(results)
```

Contact: alexjfoote@icloud.com
