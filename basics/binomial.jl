function BinomialVariate(n::Int64, p::Float64)
    head = 0
    for i = 1:n
        if rand() < p
            head += 1
        end
    end
    return head
end

BinomialVariate(10, 0.5)

data = map((x)->BinomialVariate(10, 0.5), 1:50)

mean(data)

var(data)

data = map((x)->BinomialVariate(10, 0.1), 1:50)

mean(data)

var(data)
