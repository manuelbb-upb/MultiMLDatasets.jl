module MultiMNISTModule

import MLDatasets
import MLDatasets: datafile, MNIST, SupervisedDataset, bytes_to_type
import MLDatasets: convert2image
import Interpolations: Constant
import GZip
import ImageTransformations: imresize
import Random: Xoshiro

include("idx_utils.jl")

Base.@kwdef struct CanvasSize
    width :: Int
    height :: Int
end

const SenerKoltunMNISTSettings = Dict{Symbol, Union{Tuple{Int, Int}, CanvasSize}}(
    :top_left_corner_left => (1,1),
    :top_left_corner_right => (7,7),
    :newimg_size => CanvasSize(28, 28),
    :tmp_size => CanvasSize(36, 36),
)

const LeftRightSettings = Dict{Symbol, Union{Tuple{Int, Int}, CanvasSize}}(
    :top_left_corner_left => (1,1),
    :top_left_corner_right => (29,1),
    :newimg_size => CanvasSize(; height=28, width=54),
    :tmp_size => CanvasSize(; height=28, width=54),
)

const DEFAULT_SETTINGS = SenerKoltunMNISTSettings

include("make_imgs.jl")

struct MultiMNIST <: SupervisedDataset
    metadata::Dict{String, Any}
    split::Symbol
    features::Array{<:Any, 3}
    targets::Matrix{UInt8}      # MNIST uses Int64, a bit excessive...
end

# Constructors for user, inspired by MNIST constructors.
function MultiMNIST(
    FT::Type, split::Symbol; 
    dir=nothing, force_recreate=false, kwargs...
)
    @assert split in [:train, :test]
    if split === :train
        IMAGESPATH = "train-images-idx3-ubyte.gz"
        LABELSPATH = "train-labels-idx1-ubyte.gz"
    else
        IMAGESPATH = "t10k-images-idx3-ubyte.gz"
        LABELSPATH = "t10k-labels-idx1-ubyte.gz"
    end
    features_path = datafile("MNIST", IMAGESPATH, dir)
    targets_path = datafile("MNIST", LABELSPATH, dir)
    multi_fpath = default_multi_path(features_path, kwargs)
    multi_tpath = default_multi_path(targets_path, kwargs)

    if !force_recreate && isfile(multi_fpath) && isfile(multi_tpath)
        return MultiMNIST(FT, split, multi_fpath, multi_tpath)
    end

    multi_features, multi_targets = generate_multi_data(features_path, targets_path; kwargs...)
    multi_fpath, multi_tpath = write_idx_files(multi_features, multi_targets, multi_fpath, multi_tpath)

    return MultiMNIST(FT, split, multi_fpath, multi_tpath)
end

MultiMNIST(; split = :train, Tx = Float32, dir = nothing, kws...) = MultiMNIST(Tx, split; dir, kws...)
MultiMNIST(split::Symbol; kws...) = MultiMNIST(; split, kws...)
MultiMNIST(Tx::Type; kws...) = MultiMNIST(; Tx, kws...)

# backbone constructor
function MultiMNIST(FT::Type, split::Symbol, features_path, targets_path)
    features = bytes_to_type(FT, last(read_idx(features_path)))
    targets = Matrix{UInt8}(last(read_idx(targets_path)))

    return MultiMNIST(
        Dict(
            "n_observations" => last(size(features)),
            "features_path" => features_path,
            "targets_path" => targets_path
        ),
        split,
        features,
        targets
    )
end

function generate_multi_data(
    fpath, tpath; kwargs...
)      
    _, fdata = read_idx(fpath)
    _, tdata = read_idx(tpath)

    # generate multi-feature image data
    return make_new_imgs(fdata, tdata; kwargs...)
end

function SenerKoltunMNIST(args...; kwargs...)
    kws = Dict{Any, Any}(kwargs...)
    get!(kws, :force_recreate, false)
    merge!(kws, SenerKoltunMNISTSettings)
    return MultiMNIST(args...; kws...)
end

function LeftRightMNIST(args...; kwargs...)
    kws = Dict{Any, Any}(kwargs...)
    get!(kws, :force_recreate, false)
    merge!(kws, LeftRightSettings)
    return MultiMNIST(args...; kws...)
end

function path_signature(kwargs) :: String
    sig = ""
    for (i, attr) in enumerate((:newimg_size, :tmp_size, :top_left_corner_left, :top_left_corner_right))
        v = get(kwargs, attr, getindex(DEFAULT_SETTINGS, attr))
        if v isa Tuple
            sig *= join(v, "x")
        else
            sig *= string(v)
        end
        if i != 4
            sig *= "_"
        end
    end
    return sig
end

function default_multi_path(path, kwargs = Dict())
    return string(path) * path_signature(kwargs) * ".multi.gz"
end

using ColorTypes: Gray, N0f8
# more or less copied from MNIST
convert2image(::Type{<:MultiMNIST}, x::AbstractArray{<:Integer}) =
    convert2image(MNIST, reinterpret(N0f8, convert(Array{UInt8}, x)))

function convert2image(::Type{<:MultiMNIST}, x::AbstractArray{T,N}) where {T,N}
    @assert N == 2 || N == 3
    x = permutedims(x, (2, 1, 3:N...))
    return Gray.(x)
end

end#module