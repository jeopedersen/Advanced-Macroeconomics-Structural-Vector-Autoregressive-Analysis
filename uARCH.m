function [Q, pValues] = uARCH(res,h)
% Computes ARCH test (Engle's ARCH test). Which tests for
% ARCH until lag = h.
% INPUT:
% res = matrix of estimated residuals
% h = maximum number of lags for ARCH test
%
% OUTPUT:
% Q = Test statistic
% pValue = p-value for Q with chi^2(h) distribution

X = res.^2;
[T,K] = size(X); 
con = ones(T,1);
Q = zeros(1,K);
pValues = zeros(1,K);

for ii = 1:K
    Y = X(:,ii);
    Xlag = lagmatrix(Y,[1:h]);
    Xreg = [con(h+1:end) Xlag(h+1:end,:)];
    Y = Y(h+1:end);
    N = length(Y);
    beta = (transpose(Xreg)*Xreg)^-1*(transpose(Xreg)*Y);
    e = Y - Xreg * beta;
    e2 = e.^2;
    varsum = sum((Y-mean(Y)).^2);
    R2 = 1-sum(e2)/varsum;
    Q(1,ii) = N*R2;
    pValues(1,ii) = chi2cdf(Q(1,ii),h,'upper');
end
end