using HDF5
using Compat

fi = 2 
cat="training"
#cat="test"
#data = readcsv("ACT$(fi)_competition_training.csv")
data = readcsv("ACT$(fi)_competition_$(cat).csv")

# 9493-2 = 9491 molecular descriptors
head = data[1,:]
mol  = convert(Array{String},data[2:end,1])
# normalize to [0, 1]
act  = convert(Array{Float32}, data[2:end,2]) ./ 11.0
des  = convert(Array{Float32, 2}, data[2:end,3:end])
maximum(des)
minimum(des)

m = length(mol)
idx = shuffle(1:m)
train_size = Int(round(m*0.75)) # 75% used as training
test_size = m - train_size
train = idx[1:train_size]
test  = idx[train_size+1:end]

h5open("ACT$(fi)_train.hdf5", "w") do h5
        n = length(head)-2
        m = train_size
        dset_data = d_create(h5, "data", datatype(Float32), dataspace(n,1,1,m))
        dset_label = d_create(h5, "label", datatype(Float32), dataspace(1,m))
        #dset_name = d_create(h5, "name", datatype(String), dataspace(1, m))
        for i = 1:m
                j = train[i]
                dset_data[:,1,1,i] = des[j,:]
                dset_label[1,i] = act[j]
       #         dset_name[:,i] = mol[j]
        end
end

h5open("ACT$(fi)_test.hdf5", "w") do h5
        n = length(head)-2
        m = test_size
        dset_data = d_create(h5, "data", datatype(Float32), dataspace(n,1,1,m))
        dset_label = d_create(h5, "label", datatype(Float32), dataspace(1,m))
        #dset_name = d_create(h5, "name", datatype(String), dataspace(1, m))
        for i = 1:m
                j = test[i]
                dset_data[:,1,1,i] = des[j,:]
                dset_label[1,i] = act[j]
       #         dset_name[:,i] = mol[j]
        end
end

