using Test

using Resnet18
using Flux
using TestImages
using Images
using PyCall
using Statistics

const torch = pyimport_conda("torch", "torch")
const torchvision = pyimport_conda("torchvision", "torchvision")

function dist(x, y) 
    d = x .- y
    d1 = d .|> abs |> maximum
    d2 = d.^2 |> sum |> sqrt
    mean1 = mean(x)
    mean2 = mean(y)
    sd1 = std(x)
    sd2 = std(y)
    @show d1 d2 mean1 sd1 mean2 sd2
    d2
end


img = testimage("mandrill")
img_arr = img |> channelview .|> Float32  # (3, 512, 512)

img_arr_croped = img_arr[:, 1:224, 1:224]
batch_m = reshape(img_arr_croped, 1, size(img_arr_croped)...)
batch_py = torch.tensor(batch_m)  # (1, 3, 224, 224)
batch = permutedims(batch_m, [3, 4, 2, 1])  # (224, 224, 3, 1)

model = Resnet18.resnet18()
model_py = torchvision.models.resnet18(pretrained=true);

state_dict = Dict{String, Array}()
for (key, arr) in model_py.state_dict()
    state_dict[key] = arr.numpy() |> copy
end

Resnet18.set_param!(model, state_dict)


@testset "test output" begin
    model_py.eval()

    out = model(batch)  # (1000, 1)
    out_py = model_py(batch_py)  # (1, 1000)    

    @test dist(vec(out), vec(out_py.detach().numpy())) <= 1e-4

    trainmode!(model)
    model_py.train();

    out = model(batch)
    out_py = model_py(batch_py)
    
    @test dist(vec(out), vec(out_py.detach().numpy())) < 1e-4
end

model = Resnet18.resnet18()
model_py = torchvision.models.resnet18(pretrained=true);

state_dict = Dict{String, Array}()
for (key, arr) in model_py.state_dict()
    state_dict[key] = arr.numpy() |> copy # break change?
end

Resnet18.set_param!(model, state_dict)

@testset "test gradient" begin
    model_py(batch_py).sum().backward()

    gs = gradient(params(model)) do
        batch |> model |> sum
    end

    grad_conv1_weight_py = permutedims(model_py.conv1.weight.grad.numpy(), [3, 4, 2, 1])[7:-1:1, 7:-1:1, :, :]
    grad_conv1_weight = gs.grads[model.conv1.weight];

    @test dist(grad_conv1_weight_py, grad_conv1_weight) < 1e-4
    # maxpool gradient is wrong on present NNlib implementation
    # use my fork to reduce its effect:
    # https://github.com/yiyuezhuo/NNlib.jl/tree/fix_maxpool_gradient
end

