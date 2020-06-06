# adapt from https://gist.github.com/RobBlackwell/10a1aeabeb85bbf1a17cc334e5e60acf

module io

using PyCall

# const pickle = PyNULL()
const torch = PyNULL()

function __init__()
    # copy!(pickle, pyimport("pickle"))
    copy!(torch, pyimport("torch"))
end

save(obj, filename) = torch.save(obj, filename)

load(filename) = torch.load(filename)

function load_state_dict(filename)
    r = load(filename)
    rd = Dict{String, Array}()
    for (key, value) in r
        rd[key] = value.numpy()
    end
    rd
end

end