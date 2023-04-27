
macro set_default(expr)
    @assert Meta.isexpr(expr, :(=), 2)
    lhs, rhs = expr.args
    return quote
        $lhs = isnothing($rhs) ? get(DEFAULT_SETTINGS, $(Meta.quot(rhs)), nothing) : $rhs
    end |> esc
end

tuplify(tup) = tup
tuplify(sx::CanvasSize) = (sx.width, sx.height)

#=
# Image Generator Algorithm

The below routine creates 2-Task MNIST data based on `fdata` and `tdata`, the **features**
and **targets** of some standard MNIST data set.
Each image in `fdata` is a matrix, and its top left pixel has the index `[1,1]`.
The pixel to its left is indexed with `[2, 1]`, the pixel below has index `[1,2]`.
The Algorithm creates new data in the following way:
* For each (left) image in `fdata`, choose a (right) image at random.
* Place them in a temporary canvas of size `tmp_size`.
  The left image is placed such that its top left corner has coordinates `top_left_corner_left`.
  The right image will have its top left corner placed at `top_left_corner_right`.
* Where the images overlap, the maximum pixel value is used.
* Finally, the temporary canvas is shrunk to size `newimg_size` via nearest neighbor interpolation.
* (Target labels are set accordingly).
=#
function make_new_imgs(fdata, tdata;
    top_left_corner_left::Union{Symbol, Nothing, NTuple{2,<:Integer}}=nothing,
    top_left_corner_right::Union{Symbol, Nothing, NTuple{2,<:Integer}}=nothing,
    newimg_size::Union{Nothing, NTuple{2,<:Integer}, CanvasSize}=nothing,
    tmp_size::Union{Nothing, NTuple{2,<:Integer}, CanvasSize}=nothing,
    kwargs...
)

    rng = Xoshiro(31415)      # reproducibly choose RHS image
    
    dim = ndims(fdata)
    @assert dim <= 3

    # permtuple = (2,1,3:dim...)
    # invpremtuple = invperm(permtuple)
    # pfdata = permutedims(fdata, permtuple)  # switch rows and columns, parsed IDX data is transposed
    pfdata = fdata
    s1, s2, len = size(pfdata)                # width, height, len
    
    @set_default tlc_l = top_left_corner_left
    @set_default tlc_r = top_left_corner_right
    @set_default _newimg_size = newimg_size
    @set_default _tmp_size = tmp_size
    newimg_size = tuplify(_newimg_size)
    tmp_size = tuplify(_tmp_size)
    
    #if tlc_l[2] >= tlc_r[2]
    if tlc_l[1] >= tlc_r[1]
        # left image must be left...swap corner coordinates
        _tlc_l = tlc_l
        tlc_l = tlc_r
        tlc_r = _tlc_l
    end

    # compute coordinates of left and right subimage, taking negative entries into account
    ltlc, lbrc = global_coordinates(tlc_l; local_img_size=(s1, s2), tmp_size)
    rtlc, rbrc = global_coordinates(tlc_r; local_img_size=(s1, s2), tmp_size)
   
    # compute “local” coordinates, i.e., the indices of the subimages to be placed in the canvas
    lloc_tlc, lloc_brc = local_coordinates(ltlc, lbrc, (s1, s2))
    rloc_tlc, rloc_brc = local_coordinates(rtlc, rbrc, (s1, s2))

    # is there overlap? If so, set lov_tlc and lov_brc to contain the local coordinates of 
    # the left image for the overlapping box and rov_tlc, rov_brc for the right image.
    has_overlap, lov_tlc, lov_brc, rov_tlc, rov_brc = overlap(ltlc, lbrc, rtlc, rbrc, lloc_tlc, lloc_brc, rloc_tlc, rloc_brc)
    new_imgs = similar(pfdata, (newimg_size..., len))
    new_targets = similar(tdata, (2, len))

    for ileft = 1:len
		iright = rand(rng, 1:len)    # choose random index for right feature img
		img_left = pfdata[:,:,ileft]
		img_right = pfdata[:,:,iright]
		img_new = zeros(eltype(img_left), tmp_size)
	    
        corner_view(img_new, ltlc, lbrc) .= corner_view(img_left, lloc_tlc, lloc_brc)
        corner_view(img_new, rtlc, rbrc) .= corner_view(img_right, rloc_tlc, rloc_brc)
        
        # overlap: maximum value
        if has_overlap
            corner_view(img_new, rtlc, lbrc) .= max.(
                corner_view(img_left, lov_tlc, lov_brc),
                corner_view(img_right, rov_tlc, rov_brc),
            )
        end

        new_imgs[:, :, ileft] .= imresize(img_new, newimg_size; method=Constant()) # nearest neighbor "interpolation"

        new_targets[1, ileft] = tdata[ileft]
        new_targets[2, ileft] = tdata[iright]
	end
    
    # return permutedims(new_imgs, invpremtuple), new_targets
    return new_imgs, new_targets
end

# Helpers

"""
    global_coordinates(tlc; local_img_size=(28,28), tmp_size=(36,36))

Compute actual top left and bottom right corner coordinates of an image slice with specified 
top left corner `tlc` (negative values allowed) and image size `local_img_size` in a global 
canvas of size `tmp_size.
"""
function global_coordinates(
    tlc;
    local_img_size=(28, 28), tmp_size=(36, 36)
)
    
    TLC = (1,1)                     # coordinates of global top left corner
    BRC = tmp_size                  # coordinates of global bottom right corner
    brc = tlc .+ local_img_size .- 1    # coordinates of subimage bottom right corner

    # project back into global constraints:
    _tlc = min.(BRC, max.(TLC, tlc))
    _brc = min.(BRC, max.(TLC, brc))
    return _tlc, _brc
end

"""
    local_coordinates(tlc, brc, orig_size)

Consider an image `img` with size `orig_size`, a slice of which should be placed in `canvas`
such that the slice starts at `canvas[tlc...]` and its bottom right corner is
`img[orig_size...] = canvas[brc...]`.
Return image space coordinate tuples `_tlc` and `_brc` such that
`img[_tlc[1]:_brc[1], _tlc[2]:_brc[2]] == canvas[tlc[1]:brc[1], tlc[2]:brc[2]]`.
"""
function local_coordinates(tlc, brc, orig_size)
    # actual width of subimage
    w = brc .- tlc .+ 1
    _tlc = orig_size .- w .+ 1
    _brc = orig_size
    return _tlc, _brc
end

function overlap(ltlc, lbrc, rtlc, rbrc, lloc_tlc, lloc_brc, rloc_tlc, rloc_brc)
    diag = lbrc .- rtlc

    lov_tlc = lloc_tlc
    lov_brc = lloc_brc

    rov_tlc = rloc_tlc
    rov_brc = rloc_brc

    has_overlap = all(diag .> 0)    # there is overlap, in the global canvas its top left corner is rtlc, and its bottom right corner is lbrc
    if has_overlap
        lov_tlc = lloc_brc .- diag
        rov_brc = rloc_tlc .+ diag
    end
    
    return has_overlap, lov_tlc, lov_brc, rov_tlc, rov_brc
end

function corner_view(arr, tlc, brc)
    return view(arr, tlc[1]:brc[1], tlc[2]:brc[2])
end