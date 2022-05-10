function [stat,excl] = SEWE(X,p,r,m)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% INPUT
% X  = data matrix
% p = number of lags
% r = rank
% OPTIONAL INPUT
% m = string of preffered model type. The standard model is H1

if ~exist('m','var')
    m = 'H1';
end
K = width(X);
beta_test = zeros(1,1);
if strcmp(m,'H*')
    beta_test = zeros(1,K+1);
elseif strcmp(m,'H1*')
    beta_test = zeros(1,K+1);
else
    beta_test = zeros(1,K);
end

alpha_test = zeros(1,K);
stat = zeros(K,1);
excl = zeros(K,1);
wexo = zeros(K,1);

for i = 1:K
    beta_test(1,i) = 1; 
    alpha_test(1,i) = 1;
    % Stationarity test
    [h0,pValue0,stat0,cValue0,mles0] = jcontest(X,r,'Bvec',{beta_test'},'model',m,'lags',p);
    stat(i,1) = pValue0;
    % Exclusion test
    [h2,pValue2,stat2,cValue2,mles2] = jcontest(X,r,'Bcon',{beta_test'},'model',m,'lags',p);
    excl(i,1) = pValue2;
    % Weak exogenity
    [h1,pValue1,stat1,cValue1,mles1] = jcontest(X,r,'Acon',{alpha_test'},'model',m,'lags',p);
    wexo(i,1) = pValue1;
    beta_test(1,i) = 0;
    alpha_test(1,i) = 0;
end
% Create table to display
T=table(stat,excl,wexo);
T.Properties.VariableNames = {'Stationarity' 'Exclusion' 'Weak exogeneity'};
format bank, display(T,'Tests');
format bank, table2latex(T,'SEWE.tex')
end

