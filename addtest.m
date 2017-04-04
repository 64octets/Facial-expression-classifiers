clear all;
close all;
load('data3.mat');

imageUint8 = imread('1.jpg');  %unit8
image1d = imageUint8(:,:,1);
imageDouble = double(image1d); %double
%imshow(imageUint8);
prof1 = reshape(imageDouble',1,[]);

imageUint8 = imread('2.jpg');  %unit8
image1d = imageUint8(:,:,1);
imageDouble = double(image1d); %double
%imshow(imageUint8);
prof2 = reshape(imageDouble',1,[]);

imageUint8 = imread('3.png');  %unit8
image1d = imageUint8(:,:,1);
imageDouble = double(image1d); %double
%imshow(imageUint8);
shala1 = reshape(imageDouble',1,[]);

imageUint8 = imread('4.png');  %unit8
image1d = imageUint8(:,:,1);
imageDouble = double(image1d); %double
%imshow(imageUint8);
shala2 = reshape(imageDouble',1,[]);

imageUint8 = imread('5.jpg');  %unit8
image1d = imageUint8(:,:,1);
imageDouble = double(image1d); %double
%imshow(imageUint8);
shala3 = reshape(imageDouble',1,[]);


imageUint8 = imread('6.jpg');  %unit8
image1d = imageUint8(:,:,1);
imageDouble = double(image1d); %double
%imshow(imageUint8);
shala4 = reshape(imageDouble',1,[]);

test_data = [test_data;prof1;prof2;shala1;shala2;shala3;shala4];
test_label = [test_label;3;3;3;3;0;0];

save('data4.mat','train_data','train_label','test_data','test_label');




