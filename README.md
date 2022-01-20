# Robustness Evaluation and Enhancement Toolbox

The Robustness Evaluation and Enhancement Toolbox (REEToolbox) provides tools for measuring and improving the robustness of ML models. REEToolbox uses adversarial transforms - data transforms that are adversarially optimised to fool a model - to generate challenging transformations of input data. For example, in the image below a transform that simulates changing the staining of a tissue image has been optimised to cause a trained model to misclassify a patch of tumorous tissue as non-tumorous. 

![An example of using an adversarial transform to modify the staining of an image](https://github.com/alexjfoote/reetoolbox/blob/main/example_image.png?raw=true)

Adversarial transforms allow you to measure robustness by evaluating the degradation in performance on the transformed data compared to the original data. Even accurate models are often highly vulnerable to basic adversarial transforms, like rotations. 

Fortunately, you can improve robustness by using adversarially optimised transforms for data augmentation during training. The table below shows the accuracy and fooling rate of models trained on a version of the PanNuke dataset (https://arxiv.org/abs/2003.10778) to classify image patches as tumorous or non-tumorous. Models were trained using no augmentations, random augmentations (the standard method of augmenting data during training, where random transform parameters are chosen), and adversarial augmentations (where the transform parameters are adversarially optimised). The augmentations used in the latter two cases were rotations, crops, zooms in and out, and a blurring transform that blurs part of an image. Five models were trained for each training condition, and the accuracy and fooling rate (a measure of robustness which computes the percentage of the time a model correctly classifies an image, then misclassifies it after it adversarially transformed) of the models were measured. The fooling rate was measured for each of the transforms used as augmentations during training, using adversarial optimisation. 

The models improved in accuracy and significantly improved in robustness when using adversarial augmentations compared to no augmentations, and adversarial augmentations also consistently improved accuracy and robustness compared to random augmentations.

<div align="center">
  
|| | | | | **Fooling Rate/%** | ||
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|**Augmentations**|**Accuracy/%**|**Rotate**|**Crop**|**Zoom In**|**Zoom Out**|**Blur**|**Average**|
|None|94.1|26.5|44.1|36.0|40.5|8.1|31.0|
|Random|94.7|10.6|19.9|20.8|9.0|4.3|12.9|
|Adversarial|95.3|6.5|17.2|17.1|5.0|2.2|9.6|
  
 </div>

## Usage
There are in-depth tutorial notebooks that demonstrate how to use the toolbox for robustness evaluation and adversarial training in the Tutorials directory, and at https://github.com/alexjfoote/reetoolbox-tutorials for easy downloading.

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
