#print working directory
pwd()

cd("$(homedir())/doc/code/julia/tutorial")

data1 = readcsv("demo_data_1.dat")

typeof(data1)

size(data1)

# double quotes do not work here, "t"! Character vs string?
data2 = readdlm("demo_data_1.dat", '\t')

typeof(data2)

size(data2)

#Regular arrays are generated! Not a list of lists. 
# Empty cells are filled with emptry strings ""
data3 = readdlm("demo_data_2.dat", '\t')

size(data3)

data3[1]

data3[1,1]

data3[1,2]

data3[1,3]

length(data3[1])

#last line
data3[end, 1:end]

#last 37 lnes
data3val=data3[end-36:end, 1:end]

writecsv("demo_out_1.csv", data3val)

writedlm("demo_out_2.tsv", data3val, '\t')


#Treat data  as strings. No automatic conversion.
data4 = readdlm("demo_data_2.dat", '\t', String)

