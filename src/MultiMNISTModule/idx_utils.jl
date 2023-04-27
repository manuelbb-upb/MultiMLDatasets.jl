#=================== IDX Utility functions ===============================#

# For a description of the "IDX" file format, see 
# http://yann.lecun.com/exdb/mnist/

# mapping of byte value to data type
const DATTYPE_DICT = Base.ImmutableDict(
    0x08 => UInt8,
    0x09 => Int8,
    0x0B => Int16,
    0x0C => Int32,
    0x0D => Float32,
    0x0E => Float64
)

const INV_DATATYPE_DICT = Base.ImmutableDict(
    UInt8 => 0x08,
    Int8 => 0x09,
    Int16 => 0x0B,
    Int32 => 0x0C,
    Float32 => 0x0D,
    Float64 => 0x0E
)

function read_idx(path)
    bytes = GZip.open(path, "r") do io
        parse_idx_io(io)
    end
    return bytes
end

function parse_idx_io(io)
    seekstart(io)

    # first 2 bytes are zero
    @assert iszero(read(io, UInt8))
    @assert iszero(read(io, UInt8))

    # next 2 bytes encode type and dimensionality
    dtype = DATTYPE_DICT[read(io, UInt8)]
    dims = Int(read(io, UInt8))

    # shape is given by Int32 values
    shape = zeros(Int32, dims)
    for i=1:dims
        shape[i] = ntoh(only(reinterpret(Int32, read(io, 4))))
    end

    # rest is data, but row-major (Julia reads column-major, hence the `reverse`)
    data_mat = Array{dtype}(undef, reverse(shape)...)
    read!(io, data_mat)

    return (dims=dims, shape=shape, dtype=dtype), ntoh.(data_mat)
end

function to_idx_bytes(arr)
    return reshape(reinterpret(UInt8, hton.(arr)), :)
end

function write_idx(outpath, arr)
    if haskey(INV_DATATYPE_DICT, eltype(arr))
        dtype = INV_DATATYPE_DICT[eltype(arr)]
        _arr = arr
    else
        if eltype(arr) <: Real
            dtype = Float64
            _arr = Array{Float64}(arr)
        else
            error("array has unsupported data type.")
        end
    end
    
    dims = UInt8(ndims(arr))
    shape = size(arr)

    try
        GZip.open(outpath, "w") do io
            write(io, 0x00)
            write(io, 0x00)
            write(io, dtype)
            write(io, dims)
            for len in reverse(shape)
                write(io, hton(Int32(len)))
            end
            write(io, to_idx_bytes(_arr))
        end
        return outpath
    catch
        @warn "Could not save newly generated data in IDX file at\n$(outpath)"
        return nothing
    end
end

function write_idx_files(features_array, targets_array, features_outpath, targets_outpath)
    @assert last(size(features_array)) == last(size(targets_array))

    fo = write_idx(features_outpath, features_array)
    to = write_idx(targets_outpath, targets_array)
    return fo, to
end