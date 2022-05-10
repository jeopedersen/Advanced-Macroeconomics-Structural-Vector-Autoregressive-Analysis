function [Q, pValue] = LBuni(res,h)
%LBUNI Summary of this function goes here
%   Detailed explanation goes here
q_test = [];
p_test = [];
for i = 1:width(res)
    % First code is to compare results with built in lbqtest
    [~,p,stat,~] = lbqtest(res(:,i),'Lags',h);
    q_test = [q_test,stat];
    p_test = [p_test, p];
end

% Actual lbqtest 
acf1 = zeros(h+1,width(res));
rho_squared = zeros(length(acf1),width(acf1));
q = [];
qq = zeros(h+1,width(res));
Q = zeros(1,width(res));
[T K] = size(res);
T = T;
pValue = zeros(1,width(res));
for j = 1:width(res)
    [xx,~,~] = autocorr(res(:,j),'NumLags',h);
    acf1(1:end,j) = xx;
    rho_squared = acf1.^2;
    rho_squared = rho_squared(2:end,:);
    
    for jj = 1:h
        qq1 = rho_squared(jj,:)/(T-jj);
        qq(jj,:) = qq1; 
    end

    Q(1,j) = T*(T+2)*sum(qq(:,j));
    %tkm
    dof = h;
    pValue(1,j) = chi2cdf(Q(1,j),dof,'upper');
    cValue = chi2inv(1-0.05,dof);
end
if q_test ~= Q
    warning('Problem, the build in function does not match the results of the test conducted manually');
else
   % do nothing
end
end