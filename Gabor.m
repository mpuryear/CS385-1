filename = '2.jpg'; 
b = imread(filename);
figure, imshow(b)
b = rgb2gray(b);

%{
wavelength = 4;
orientation = 90;
[mag,phase] = imgaborfilt(b,wavelength,orientation);
imshow(phase,[]);
%}

gaborArray = gabor([4 8],[0 90]);
gaborMag = imgaborfilt(b,gaborArray);
figure
subplot(2,2,1);
for p = 1:4
    subplot(2,2,p)
    imshow(gaborMag(:,:,p),[]);
    theta = gaborArray(p).Orientation;
    lambda = gaborArray(p).Wavelength;
    title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
end