function AR1TimeSeries(a::Float64; x0::Float64 = 0.0, T::Int64 = 200)
    x = x0
    result = [x]
    for i = 1:T
        x = a*x + rand()
        push!(result, x)
    end
    return result
end

AR1TimeSeries(0.0)

AR1TimeSeries(0.5)

plot(AR1TimeSeries(0.9), Geom.point, x="time", y="value")

