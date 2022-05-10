function [Beta,CovBeta,tratioBeta,res,indep,so]=VARls(y,p,con,tr);
% Input:
% y = T*K matrix
% p = number of lags
% con = 1 add constant otherwise no constant
% tr = 1 add linear trend otherwise no trend
%
% Output:
% Beta = coefficient estimates (Note: Constant and trend last in matrix)
% SEBeta = standard error of coefficients
% tratioBeta = t-ratios
% res = residuals txK matrix
% indep = matrix of independent variables
% so = variance-covariance matrix of residuals
%
%
% Define dependent and independent variables for VAR estimation
%
% Dependent variable
[T,K] = size(y);
dep = y(p+1:length(y),:);
% Independent variable
indep = lagmatrix(y,1:p); 
indep = indep(p+1:length(indep),:);
% Adding constant
if con==1;
 indep = [indep ones(length(indep),1)];
end 
% Adding linear trend
if tr==1;
 indep = [indep (1:length(indep))'];
end
% Note: we need to find the number of parameters in each of the K equations
% in order to make the small sample correction of the covariance matrix
% we also need the number of obs
[T,Kp]=size(indep);
% beta = inv(indep'*indep)*indep'*dep; which is equivalent to
Beta = indep\dep;
res = dep-indep*Beta;
so = (res'*res)/(T-Kp);
CovBeta = kron(so,inv(indep'*indep));
% we need to vectorize beta to make it consistent with its covariance
% matrix in order to compute the t-ratios
tratioBeta = reshape(Beta, [Kp*K,1])./sqrt(diag(CovBeta));
end
