```@meta
EditURL = "<unknown>/docs/src/literate/multi_mnist.jl"
```

# Default 2-Task MNIST data

Load the MultiMNIST training data formatted similarly to how its done
by Sener & Koltun.
The `force_recreate` flag ensures, that fresh IDX files are written.

````@example multi_mnist
using MultiMLDatasets
mmnist = SenerKoltunMNIST(; force_recreate=true)
````

We can inspect our data

````@example multi_mnist
using Plots
function plot_mmnist(mmnist, i)
    img = convert2image(mmnist, i)
    labels = Int.(mmnist[i].targets)
    plot(img; title=labels, aspect_ratio=:equal)
end
plot_mmnist(mmnist, 1)
````

The test set is separate:

````@example multi_mnist
mmnist_test = SenerKoltunMNIST(:test; force_recreate=true)
plot_mmnist(mmnist_test, 1)
````

# Easier tasks:

````@example multi_mnist
mmnist = LeftRightMNIST(; force_recreate=true)
plot_mmnist(mmnist, 1)
mmnist_test = LeftRightMNIST(:test; force_recreate=true)
plot_mmnist(mmnist_test, 1)
````

# Custom Format
The keywoard arguments
* `top_left_corner_left`
* `top_left_corner_right`
* `newimg_size`
* `tmp_size`
are passed down to `make_new_imgs`.

That routine creates 2-Task MNIST data based on `fdata` and `tdata`, the **features**
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

`newimg_size` and `tmp_size` can be provided as tuples, but this is rather confusing:
We think in x-y-coordinates to create the images, but `newimg_size=(10, 20)` will create
images that have a height of 10 and a width of 20 pixels, in contrast to how we think
coordinates.
Better use `CanvasSize(;width=10, height=20)`.

````@example multi_mnist
# move left image more left
mmnist = MultiMNIST(;
    force_recreate=true, top_left_corner_left=(-10, 2), top_left_corner_right=(20, 8)
)
plot_mmnist(mmnist, 2)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

