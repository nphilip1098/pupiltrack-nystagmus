% An Adaptive Threshold Algorithm based on Cumulative Histogram

I = imread('pupil.PNG');
imshow(I);
[counts,binLocations] = imhist(I);
plot(counts)
pdf = counts/ sum(counts);
cdf = cumsum(pdf);
Y = prctile(cdf,[0 10 20 30 40 50 60 70 80 90 100],'all')
fx = gradient(Y)
disp(length(fx));

for i = 1:length(fx)
    if(fx(i)>fx(i+1))
        disp(i)
        break
    end
end
disp(fx(i))
disp(Y(i))

subplot(3, 1, 1);
plot(fx);
title("Gradient of Inverse CDF")
grid("on")

subplot(3, 1, 2);
plot(Y);
title("Inverse CDF")
grid("on")

subplot(3, 1, 3);
plot(cdf);
title("CDF")
grid("on")
