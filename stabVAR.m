function [e]=stabVAR(A)
% Check stability of VAR model
% Input: A = KpxKp companion matrix
% Output: e = eigenvalues
%
% Michael Bergman December 2018
%
e = sort(abs(eig(A)),'descend');
display(e,'Eigenvalue stability condition')
end