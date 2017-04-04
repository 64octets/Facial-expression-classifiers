%{
feature = csvread('fer20130.csv',0,1);
label = csvread('fer20130.csv',0,0,[0,0,10,0]);

for i = 1:10
face1row = feature(i,:);
face = reshape(face1row,[48,48]);
man1 = uint8(face');
figure
imshow(man1);
end
%}
figure
x = 2:2:20;
y = [0.6453    0.5961    0.6552    0.6478    0.6576    0.6847    0.6650    0.6355    0.6207    0.6675];
plot(x,y,'b-o');
