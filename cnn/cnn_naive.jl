pwd()

ENV["MOCHA_USE_NATIVE_EXT"] = "true"
ENV["OMP_NUM_THREADS"] = 4
using Mocha

data_layer  = HDF5DataLayer(name="train-data", 
                            source="train.txt",
                            batch_size=100, 
                            shuffle=true)
conv_layer = ConvolutionLayer(name="conv1", 
                              n_filter=20, 
                              kernel=(25,1),
                              bottoms=[:data], tops=[:conv1])
pool_layer = PoolingLayer(name="pool1", 
                          kernel=(4,1), 
                          stride=(4,1),
                          bottoms=[:conv1], tops=[:pool1])
conv2_layer = ConvolutionLayer(name="conv2", 
                               n_filter=50, 
                               kernel=(25,1),
                               bottoms=[:pool1], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", 
                           kernel=(4,1), 
                           stride=(4,1),
                           bottoms=[:conv2], tops=[:pool2])
fc1_layer  = InnerProductLayer(name="ip1", 
                               output_dim=50,
                               neuron=Neurons.Tanh(), 
                               bottoms=[:pool2], tops=[:ip1])
fc2_layer  = InnerProductLayer(name="ip2", 
                               output_dim=1,
                               bottoms=[:ip1], tops=[:ip2])
loss_layer = SquareLossLayer(name="loss", 
                              bottoms=[:ip2,:label])
common_layers = [conv_layer, pool_layer, 
                 conv2_layer, pool2_layer,
                 fc1_layer, fc2_layer]

backend = CPUBackend()
init(backend)
net = Net("merck-train", 
          backend, 
          [data_layer, common_layers..., loss_layer])

exp_dir = "snapshots"
method = SGD()
params = make_solver_parameters(method, 
                                max_iter=1000, 
                                regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                load_from=exp_dir)
solver = Solver(method, params)
# every 1000 iterations, save the statistics to disk
setup_coffee_lounge(solver, 
                    save_into="$exp_dir/statistics.jld", 
                    every_n_iter=1000)
# every 100 iterations, show the training summary
add_coffee_break(solver, 
                 TrainingSummary(), 
                 every_n_iter=50)
# every 500 iterations, save the snapshot
add_coffee_break(solver, 
                 Snapshot(exp_dir), 
                 every_n_iter=5000)
# Performace test network 
# test data
data_layer_test = HDF5DataLayer(name="test-data", 
                                source="test.txt", 
                                batch_size=100)
acc_layer = SquareLossLayer(name="test-accuracy", 
                        bottoms=[:ip2, :label])
test_net = Net("MNIST-test", 
               backend, 
               [data_layer_test, common_layers..., acc_layer])
# every 500 iterations, performance validation on test data
add_coffee_break(solver, 
                 ValidationPerformance(test_net), 
                 every_n_iter=500)

# let's solve it!
solve(solver, net)

dump_statistics(solver.coffee_lounge. get_layer_state(net, "loss", true))

# free memory
destroy(net)
destroy(test_net)
shutdown(backend)

