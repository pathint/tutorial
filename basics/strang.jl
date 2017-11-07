# Strang Matrix Creation

function Strang(n::Int64)
    a = Array{Int64, 2}(n, n)
    for i = 1:n
        for j = 1:n
            if i==j
                a[i, j] = -2
            elseif abs(i-j) == 1
                a[i, j] = 1
            else
                a[i, j] = 0
            end
        end
    end
    return a
end

Strang(5)

