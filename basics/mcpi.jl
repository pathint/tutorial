# Implementation 1: fixed number of trials

function MonteCarloPi(n::Int64 = 10000)
    count = 0;
    for i = 1:n
        x,y = rand(2)
        if (x-0.5)^2+(y-0.5)^2 < 0.25 
            count += 1
        end
    end
    MyPi = 4*count/n
    println("Pi estimated from ", n, " random trials is ", MyPi)
    println("With an absolute error of ", MyPi - π)
    return MyPi
end

# Test
MonteCarloPi(10000)

MonteCarloPi()

# Implementation 2: fixed convergence critetion
function MonteCarloPi(ϵ::Float64)
    count = 0
    n = 0
    MyPi = 0
    while abs(π - MyPi) > abs(ϵ)
        n += 1
        x,y = rand(2)
        if (x-0.5)^2+(y-0.5)^2 < 0.25 
            count += 1
        end
        MyPi = 4*count/n
    end
    println("Pi estimated from ", n, " random trials is ", MyPi)
    println("With an absolute error of ", MyPi - π)
    return MyPi
end

MonteCarloPi(1.0e-1)

MonteCarloPi(1.0e-2)

MonteCarloPi(1.0e-3)

MonteCarloPi(1.0e-4)

MonteCarloPi(1.0e-5)

