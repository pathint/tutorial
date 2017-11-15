cd("$(homedir())/doc/code/julia/ml/wisconsin_breast_cancer")

pwd()

data = readcsv("data.csv")

# column 2: diagnosis: M = malignant, B = benign

function tobin(x)
    if x=="B"
        0
    else
        1
    end
end

# convert into integer format
label = map(tobin,  data[2:end, 2])

# 357 benign samples
count(x-> x == 0, label)

# 212 malignant samples
count(x-> x == 1, label)

# columns: 3:32 measured indices  
# convert to Float64 two-dimensional array
obs = convert(Matrix{Float64}, data[2:end, 3:end-1])

# do a random shufffle 
order = shuffle(1:length(label))

#take the first 300 as train set, the rest as test set
train = order[1:300]
test = order[301:end]

#count how many Bs and Ms in the train set
count(x-> x == 0, label[train])

count(x-> x == 1, label[train])

using LIBSVM

# train the model
model = fit!(LinearSVC(), # initialize model, could be SVC, NuSVC, LinearSVC, NuSVR 
             obs[train, 1:end]', # features
             label[train]) # labels

# predict on the test 
pred = predict(model, obs[test, 1:end]')

length(test)

# Number of correct predictions
count(x->x==true, map(==, pred, label[test]))


