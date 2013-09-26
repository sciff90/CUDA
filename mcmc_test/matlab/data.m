fid = fopen('../data/a.dat');
A = fscanf(fid,'%f',[1 inf]);
A = A';
A( ~any(A,2), : ) = [];  %rows
%A( :, ~any(A,1) ) = [];  %columns
fclose(fid);

fid = fopen('../data/b.dat');
B = fscanf(fid,'%f',[1 inf]);
B = B';
B( ~any(B,2), : ) = [];  %rows
%B( :, ~any(B,1) ) = [];  %columns
fclose(fid);

figure(1)

hist(A(:),1000);
figure(2)

hist(B(:),1000);


