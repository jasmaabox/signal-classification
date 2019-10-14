c = {
    reshape(1:(4*4*3), [4, 4, 3]),
    ones(4, 4, 3),
};

A = cell2mat(c);
A;

A = permute(A, [3, 2, 1]);
B = reshape(A, [3, 4, 4, 2]);
B = permute(B, [4, 3, 2, 1]);

cell2mat(c(1)) == squeeze(B(1,:,:,:))