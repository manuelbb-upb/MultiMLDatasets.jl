module MultiMLDatasets

include("MultiMNISTModule/MultiMNISTModule.jl")
import .MultiMNISTModule: MultiMNIST, SenerKoltunMNIST, convert2image, LeftRightMNIST, CanvasSize

export MultiMNIST, SenerKoltunMNIST, convert2image, LeftRightMNIST, CanvasSize
end
