# Resnet18 implemented in Flux but use PyTorch pretrained weights

## Create a Flux Resnet using existing state_dict

```julia
using Resnet18

model = Resnet18.resnet18()
state_dict = load_state_dict("resnet18.pkl")
Resnet18.set_param!(model, state_dict)
```

## Create a PyTorch Resnet18 and borrow its weight

```julia

using Resnet18
using Flux
using PyCall

const torch = pyimport_conda("torch", "torch")
const torchvision = pyimport_conda("torchvision", "torchvision")

model = Resnet18.resnet18()
model_py = torchvision.models.resnet18(pretrained=true);

state_dict = Dict{String, Array}()
for (key, arr) in model_py.state_dict()
    state_dict[key] = arr.numpy() |> copy
end

Resnet18.set_param!(model, state_dict)
```

## Pitfall

The PyTorch's `Conv` runs a more intuitive "Correlation" in fact while Flux's `Conv` runs traditional convolution exactly. Fortunately, there's not a huge gap. If you apply `[end:-1:start, end:-1:start, 1, 1]` transform to weight, they're equivalent to each other.

Present Flux(NNlib) uses wrong gradient implementation for `MaxPool` (see this [issue](https://github.com/FluxML/NNlib.jl/issues/205)). Use my [fork](https://github.com/yiyuezhuo/NNlib.jl/tree/fix_maxpool_gradient_further) to fix it. If the gradient is wrong, the gradient check included in the package unit test will fail to pass.
