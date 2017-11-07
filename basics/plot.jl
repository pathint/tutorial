using Gadfly


plot(x=rand(10), y=rand(10))

plot(x=rand(10), y=rand(10), Geom.point, Geom.line)

plot(x=rand(10), y=rand(10), Geom.point, Geom.smooth)

plot(x=1:10, y=2.^rand(10),
     Scale.y_sqrt, Geom.point, Geom.smooth,
     Guide.xlabel("Stimulus"), Guide.ylabel("Response"), Guide.title("Dog Training"))

img1 = plot(x=1:10, y=exp2.(rand(10)),
     Scale.y_log2, Geom.point, Geom.smooth,
     Guide.xlabel("Stimulus"), Guide.ylabel("Response"), Guide.title("Dog Training"))

draw(PDF("test_plot1.pdf", 4inch, 3inch), img1)

plot(sin, -2π, 2π) 

plot([sin, cos], -2π, 2π)

draw(PDF("test_plot2.pdf", 4inch, 3inch), plot([sin, cos], -2π, 2π)) 


