function [Q_h, pValue, mQ_h, mpValue] = Portmanteau(res,h,p)
% Computes Portmanteau test (Multivariate Ljung-Box test). Which tests for
% autocorrelation until lag = lags.
% INPUT:
% res = matrix of estimated residuals
% h = maximum number of lags for Portmanteau test
% p = number of lags in estimated VAR(p), used for calculating degrees
% of freedom
%
% OUTPUT:
% Q_h = Test statistic (2.6.1)
% pValue = p-value for Q_h with chi^2(K^2(h-p)) distribution
%
% NOTE:
% The number h should be considerably larger than p for a good
% approximation to the null distribution. Choosing h too large, however,
% may undermine the power of the test.
[T K] = size(res);
C_0 = T^-1*(res'*res);
dof = K^2*(h-p);

%Portmanteau
Q = [];
tt = 1;
while tt < h + 1
    lagged_m = lagmatrix(res,tt);
    C_j = T^-1 *(transpose(res(tt+1:end,:))*lagged_m(tt+1:end,:));
    Q(tt) = trace(C_j'*inv(C_0)*C_j*inv(C_0));
    tt = tt + 1;
end
Q_h = T*sum(Q)
pValue = chi2cdf(Q_h,dof,'upper');

% Modified Portmanteau
Q_o = zeros(h,1);
oo = 1;
while oo < h + 1
    lagged_m = lagmatrix(res,oo);
    C_j = T^-1 *(transpose(res(oo+1:end,:))*lagged_m(oo+1:end,:));
    Q_o(oo) = 1/(T-oo)*trace(C_j'*inv(C_0)*C_j*inv(C_0));
    oo = oo + 1;
end
mQ_h = T^2*sum(Q_o)
mpValue = chi2cdf(mQ_h,dof,'upper');
Q_out = [Q_h mQ_h];
p_out = [pValue mpValue];
dof_out = [dof dof];
test = [Q_out;p_out;dof_out];
disp(table(categorical({'test statistic' ; 'p-value' ; 'degrees of freedom'}),test(:,1), test(:,2),'VariableNames',{'Test' 'Portmanteau' 'Modified Portmanteau'}));
tableforlatex = table(({'test statistic' ; 'p-value' ; 'degrees of freedom'}),test(:,1), test(:,2),'VariableNames',{'Test' 'Portmanteau' 'Modified Portmanteau'});
table2latex(tableforlatex,'mPortmanteau.tex')
end
