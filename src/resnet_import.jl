using Logging

function copyVec!(dst::AbstractVector, src::AbstractVector)
    # https://github.com/JuliaLang/julia/commit/d753901ad6c5ccab68d5cd4939761c1ba9caa1ea
    # This fix have not been released so I export it there
    if length(dst) != length(src)
        resize!(dst, length(src))
    end
    for i in eachindex(dst, src)
        @inbounds dst[i] = src[i]
    end
    dst
end

set_param!(modul, dict::Dict) = set_param!(modul, dict, String[])

function set_param!(modul, dict::Dict, prefixs::Vector{String})
    @debug prefixs
    p_map = Flux.trainable(modul) # named tuple
    for (key, m) in zip(keys(p_map), p_map)
        set_param!(m, dict, [prefixs; [String(key)]])
    end
end

function set_param!(modul::Conv, dict::Dict, prefixs::Vector{String})
    @debug prefixs
    weight_key = join([prefixs; ["weight"]], ".")
    bias_key = join([prefixs; ["bias"]], ".")
    
    w = dict[weight_key]
    w = permutedims(w, [3,4,2,1]) # align to pytorch original order
    w = w[size(w,1):-1:1, size(w,2):-1:1, :, :]  # convert correlation kernel to conv kernel
    @debug "dummy set: $(modul.weight |> size) <- dict: $(w |> size)"
    copy!(modul.weight, w)
    
    if bias_key in keys(dict)
        @debug "dummy set: $(modul.bias |> size) <- dict: $(dict[bias_key] |> size)"
        copyVec!(modul.bias, dict[bias_key])
    else
        # Flux last release have not add bias free Conv
        # https://github.com/FluxML/Flux.jl/pull/873/files/a4a987f0b0c7745a05f4322eaaa87f422ce990b6
        @debug "Missing bias"
    end
end

function set_param!(modul::BatchNorm, dict::Dict, prefixs::Vector{String})
    @debug prefixs
    running_mean_key = join([prefixs; ["running_mean"]], ".")
    running_var_key = join([prefixs; ["running_var"]], ".")

    weight_key = join([prefixs; ["weight"]], ".")
    bias_key = join([prefixs; ["bias"]], ".")
    
    @debug "dummy set: $(modul.μ |> size) <- dict: $(dict[running_mean_key] |> size)"
    #println("$(modul.μ) \n\n $(dict[running_mean_key])")
    copyVec!(modul.μ, dict[running_mean_key])
    @debug "dummy set: $(modul.σ² |> size) <- dict: $(dict[running_var_key] |> size)"
    copyVec!(modul.σ², dict[running_var_key])

    copyVec!(modul.γ, dict[weight_key])
    copyVec!(modul.β, dict[bias_key])
end

function set_param!(modul::Chain, dict::Dict, prefixs::Vector{String})
    @debug prefixs
    for (i, layer) in enumerate(modul.layers)
        set_param!(layer, dict, [prefixs; [string(i-1)]])
    end
end

function set_param!(modul::Dense, dict::Dict, prefixs::Vector{String})
    @debug prefixs
    weight_key = join([prefixs; ["weight"]], ".")
    bias_key = join([prefixs; ["bias"]], ".")
    
    @debug "dummy set $(modul.W |> size) <- $(dict[weight_key] |> size)"
    copy!(modul.W, dict[weight_key])
    @debug "dummy set $(modul.b |> size) <- $(dict[bias_key] |> size)"
    copyVec!(modul.b, dict[bias_key])
end
