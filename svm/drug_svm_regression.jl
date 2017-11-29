cd("$(homedir())/doc/code/julia/ml/merck_activity")

GroupIndex = 1
TrainSet = readcsv(string("TrainingSet/ACT", GroupIndex, "_competition_training.csv"))

Head = TrainSet[1, 1:end] # Head
Molecule = TrainSet[2:end, 1] # Molecule ID
Activity = convert(Vector{Float64}, TrainSet[2:end, 2]) # Activity
Features = convert(Matrix{Float64}, TrainSet[2:end, 3:end]) # Molecular Descriptors
clear!(:TrainSet) # Free Memory
                   
using LIBSVM

# train the model
@time model = fit!(NuSVR(), # initialize model, could be SVC, NuSVC, LinearSVC, NuSVR 
             Features', # features
             Activity) # activity

# predict on the test 
# TestSet/ACT1_competition_test.csv
TestSet = readcsv(string("TestSet/ACT", GroupIndex, "_competition_test.csv"))

TestFeatures = convert(Matrix{Float64}, TestSet[2:end, 2:end])

@time TestPred = predict(model, TestFeatures')

# Number of correct predictions
count(x->x==true, map(==, pred, label[test]))


