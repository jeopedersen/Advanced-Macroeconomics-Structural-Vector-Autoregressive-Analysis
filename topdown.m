function [] = topdown(y,pmax)
%-------------------------------------------------------%
% Specifying the number of lags with top-down sequential testing
% Redefining the VAR 

% Constructing regressor matrix and dependent variable
yt=y(pmax+1:end,:); 
ylags = lagmatrix(y,1:pmax); 
ylags = ylags(pmax+1:length(ylags),:);
ylags = [ ones(length(ylags),1) ylags ];

% Computing sample, t, and number of equations, K
[t,K] = size(yt);   

% Estimating the VAR(m) and the VAR(m+1) models
% Now we can estimate the VAR(m) and the VAR(m+1) models
% Compute all log-likelihoods
LL = [];
ii = pmax;
while ii>-1
yl=ylags(:,1:ii*K+1);
Beta = yl\yt;
res = yt-yl*Beta;
sopmax = (res'*res)/(t);
% Log-Likelihood: Hamilton (1994);
LL = [ ii -(t/2)*(log(det(sopmax))+K*log(2*pi)+K); LL ];
ii=ii-1;
end

% Set up table with results
% Note LR-test: LR(p) = 2*(LL(p)-LL(p-1))

results = [ 0 LL(1,2) 0 0];

ii=1;
while ii<pmax+1
results = [results; LL(ii+1,1) LL(ii+1,2) 2*( LL(ii+1,2) - LL(ii,2) ) 1-chi2cdf(2*( LL(ii+1,2) - LL(ii,2) ),K^2)];
ii=ii+1;
end
disp('Top-Down Testing Sequence');
disp(table(results(:,1),results(:,2),results(:,3),results(:,4),'VariableNames',{'Lag' ; 'Log. Likelihood' ; 'LR test' ; 'p-value'}));
t = table(results(:,1),results(:,2),results(:,3),results(:,4),'VariableNames',{'Lag' ; 'Log. Likelihood' ; 'LR test' ; 'p-value'});
table2latex(t,'TD.tex');
end

