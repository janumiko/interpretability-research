from torchvision.models.resnet import (
    ResNet50_Weights,
    _resnet,
    Bottleneck,
)
import torch
from modified_bottleneck import ModifiedBottleneck


# real torchiviosn resnet50
real_model = _resnet(Bottleneck, [3, 4, 6, 3], ResNet50_Weights.IMAGENET1K_V2, True)
state_dict = real_model.state_dict()

layer_names = {f"{layer}{i}" for i in range(1, 4) for layer in ["conv", "bn"]}
for key in list(state_dict.keys()):
    parts = key.split(".")
    if parts[-2] in layer_names and key.startswith("layer"):
        parts.insert(-2, "sequential_block")
        new_key = ".".join(parts)
        state_dict[new_key] = state_dict.pop(key)


model = _resnet(ModifiedBottleneck, [3, 4, 6, 3], weights=None, progress=True)
model.load_state_dict(state_dict)

# test the models with a random imagenet-like input

random_input = torch.rand(1, 3, 224, 224)
model.eval()
real_model.eval()

print(torch.allclose(model(random_input), real_model(random_input)))
