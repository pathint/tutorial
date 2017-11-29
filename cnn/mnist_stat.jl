using Mocha
backend = CPUBackend()
init(backend)

mem_data = MemoryDataLayer(name = "data",
                           tops = [:data],
                           batch_size = 1,
                           data = Array[zeros(Float32, 28, 28, 1, 1)])
softmax_layer = SoftmaxLayer(name = "prob",
                             tops = [:prob],
                             bottoms = [:ip2])


# Previous definition on the net structure 
conv_layer = ConvolutionLayer(name="conv1", 
                              n_filter=20, 
                              kernel=(5,5),
                              bottoms=[:data], tops=[:conv1])
pool_layer = PoolingLayer(name="pool1", 
                          kernel=(2,2), 
                          stride=(2,2),
                          bottoms=[:conv1], tops=[:pool1])
conv2_layer = ConvolutionLayer(name="conv2", 
                               n_filter=50, 
                               kernel=(5,5),
                               bottoms=[:pool1], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", 
                           kernel=(2,2), 
                           stride=(2,2),
                           bottoms=[:conv2], tops=[:pool2])
fc1_layer  = InnerProductLayer(name="ip1", 
                               output_dim=500,
                               neuron=Neurons.ReLU(), 
                               bottoms=[:pool2], tops=[:ip1])
fc2_layer  = InnerProductLayer(name="ip2", 
                               output_dim=10,
                               bottoms=[:ip1], tops=[:ip2])
common_layers = [conv_layer, pool_layer,
                 conv2_layer, pool2_layer,
                 fc1_layer, fc2_layer]

#define the CNN 
run_net = Net("imagenet", backend,
              [mem_data, common_layers..., softmax_layer])

#load saved model
load_snapshot(run_net, "snapshots/snapshot-010000.jld")

#read a test data, print its label
using HDF5

h5open("data/test.hdf5") do f
    get_layer(run_net, "data").data[1][:,:,1,1] = f["data"][:,:,1,1]
    println("correct label index: ", Int64(f["label"][:,1][1]+1))
end

forward(run_net)
println("Label probability vector:")
run_net.output_blobs[:prob].data

#read saved statistics data
using JLD
stats = load("snapshots/statistics.jld")

tables = stats["statistics"]
ov = tables["obj_val"]
xy = sort(collect(ov))

x = [i for (i,j) in xy]
y = [j for (i,j) in xy]

using Gadfly

plot(x=x, y=y, 
     Geom.line,  
     Guide.xlabel("Iterations"), Guide.ylabel("Objective Value"))

# low-filter smoothing
function low_pass{T <: Real}(x::Vector{T}, window::Int)
    len = length(x)
    y = Vector{Float64}(len)
    for i in 1:len
        lo = max(1, i - window)
        hi = i
        y[i] = mean(x[lo:hi])
    end
    return y
end

window = Int64(round(length(xy)/4.0))
y_avg = low_pass(y, window)
plot(x=x, y=y_avg, 
     Geom.line,  
     Guide.xlabel("Iterations"), Guide.ylabel("Objective Value"))

plot(
     layer(x=x, y=y, Geom.line, Theme(default_color=colorant"red")),  
     layer(x=x, y=y_avg, Geom.line, Theme(default_color=colorant"blue")),  
     Guide.xlabel("Iterations"), Guide.ylabel("Objective Value"))
