using Flux
using Flux: @functor

function conv3x3(in_planes, out_planes; stride=1)
    Conv((3,3), in_planes => out_planes; pad=1, stride=stride)
end

function conv1x1(in_planes, out_planes; stride=1)
    Conv((1,1), in_planes => out_planes; pad=0, stride=stride)
end

struct BasicBlock
    conv1::Conv
    bn1::BatchNorm
    conv2::Conv
    bn2::BatchNorm
end

@functor BasicBlock

function BasicBlock(inplanes, planes; stride=1, base_width=64, dilation=1)
    BasicBlock(
        conv3x3(inplanes, planes; stride=stride),
        BatchNorm(planes, relu),
        conv3x3(planes, planes),
        BatchNorm(planes)
    )
end

function (m::BasicBlock)(x)
    y = x |> m.conv1 |> m.bn1 |> m.conv2 |> m.bn2
    relu.(x + y)
end

struct BasicBlockDownsample
    conv1::Conv
    bn1::BatchNorm
    conv2::Conv
    bn2::BatchNorm
    downsample::Chain
end

@functor BasicBlockDownsample

function BasicBlockDownsample(inplanes, planes; stride=1, base_width=64, dilation=1, downsample)
    BasicBlockDownsample(
        conv3x3(inplanes, planes; stride=stride),
        BatchNorm(planes, relu),
        conv3x3(planes, planes),
        BatchNorm(planes),
        downsample
    )
end

function (m::BasicBlockDownsample)(x)
    y = x |> m.conv1 |> m.bn1 |> m.conv2 |> m.bn2
    relu.(m.downsample(x) + y)
end

# ignore Bottlenect module

function make_layer(inplanes::Int, planes::Int, blocks::Int; base_width=64, stride=1)
    downsample=nothing
    layers = Any[]
    if stride > 1
        downsample = Chain(conv1x1(inplanes, planes; stride=stride),
                           BatchNorm(planes))
        push!(layers, BasicBlockDownsample(inplanes, planes; stride=stride, base_width=base_width, downsample=downsample))
    else
        push!(layers, BasicBlock(inplanes, planes; stride=stride, base_width=base_width))
    end
    
    for i in 2:blocks
        push!(layers, BasicBlock(planes, planes; base_width=base_width))
    end
    Chain(layers...)
end

Base.@kwdef struct ResNet
    conv1::Conv
    bn1::BatchNorm
    maxpool::MaxPool
    layer1::Chain
    layer2::Chain
    layer3::Chain
    layer4::Chain
    avgpool::GlobalMeanPool
    fc::Dense
end

@functor ResNet

function ResNet(layers::Vector{Int}, num_classes::Int)
    ResNet(;
        conv1 = Conv((7,7), 3 => 64; stride=2, pad=3),
        bn1 = BatchNorm(64, relu),
        maxpool = MaxPool((3, 3); stride=2, pad=1),
        layer1 = make_layer(64, 64, layers[1]),
        layer2 = make_layer(64, 128, layers[2]; stride=2),
        layer3 = make_layer(128, 256, layers[3]; stride=2),
        layer4 = make_layer(256, 512, layers[4]; stride=2),
        avgpool = GlobalMeanPool(),
        fc = Dense(512, num_classes)
    )
end

function(m::ResNet)(x)
    x |> m.conv1 |> m.bn1 |> m.maxpool |> m.layer1 |> m.layer2 |> m.layer3 |> m.layer4 |> 
        m.avgpool |> flatten |> m.fc
end

resnet18() = ResNet([2, 2, 2, 2], 1000)

function Flux.testmode!(m::ResNet, mode = true)
    for m in [m.bn1, m.layer1, m.layer2, m.layer3, m.layer4]
        Flux.testmode!(m, mode)
    end
end

function Flux.testmode!(m::BasicBlock, mode = true)
    Flux.testmode!(m.bn1, mode)
    Flux.testmode!(m.bn2, mode)
end

function Flux.testmode!(m::BasicBlockDownsample, mode = true)
    Flux.testmode!(m.bn1, mode)
    Flux.testmode!(m.bn2, mode)
    Flux.testmode!(m.downsample, mode)  # testmode! for a BN layer which is located on Chain
    # downsample_bn = m.downsample.layers[2]
    # Flux.testmode!(downsample_bn)
end
