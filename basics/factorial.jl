function fact(n::Int64)
    result = 1
    for i  = 1:n
        result *= i
    end
    return result
end


map(factorial, 10:20) 

map(factorial, 10:25) 

map(fact, 10:20) 

map(fact, 10:25) 

##Overflow
fact(30)



