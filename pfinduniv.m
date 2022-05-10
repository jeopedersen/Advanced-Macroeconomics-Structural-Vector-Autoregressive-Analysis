function [infomat,sichat,hqchat,aichat]=pfinduniv(y,pmax);
% Computes AIC, HQ and SC information criteria for VAR models
% for p=1,2,..,pmax using the same number of observations for
% all models.
%
% Input:
% y = TxK time series vectors
% pmax = maximum number of lags
%
% Output: 
% infomat = matrix containing all results
% lag order p found using SC (sichat), HQ (hqchat) and AIC (aichat)
% 
% Lutz Kilian
% University of Michigan
% November 2008
% 
% Output added: Michael Bergman 2018
[t,K]=size(y);
% Construct regressor matrix and dependent variable
XMAX=ones(1,t-pmax);
for i=1:pmax
	XMAX=[XMAX; y(pmax+1-i:t-i,:)'];
end;
Y=y(pmax+1:t,:)';   
% Evaluate criterion for p=0,...,pmax
for jj=0:pmax
	X=XMAX(1:jj*K+1,:);
	B=Y*X'*inv(X*X');        
    SIGMA=(Y-B*X)*(Y-B*X)'/(t-pmax);
	np=length(vec(B));      % Number of freely estimated parameters
   % Lutkepohl suggests np=length(vec(B))-K which is used in table 4.5.
   % This does not affect the ranking, but the value of the criterion
   % function. See p. 147.
    aiccrit(jj+1,1)=log(det(SIGMA))+np*2/(t-pmax);    	   	         % AIC value
    hqccrit(jj+1,1)=log(det(SIGMA))+np*2*log(log(t-pmax))/(t-pmax);  % HQC value
    siccrit(jj+1,1)=log(det(SIGMA))+np*log(t-pmax)/(t-pmax);         % SIC value
end

infomat = [ siccrit hqccrit aiccrit ];
% Rank models for p = 0,1,2,...,pmax
[critmin,critorder]=min(aiccrit);
aichat=critorder-1;

[critmin,critorder]=min(hqccrit);
hqchat=critorder-1;

[critmin,critorder]=min(siccrit);
sichat=critorder-1;
% Print table with results
m = [0:pmax]';
% Add first column to results
imat = [m infomat];
% Print Table
T = table(imat(:,1),imat(:,2),imat(:,3),imat(:,4));
T.Properties.VariableNames = {'p' 'SIC' 'HQC' 'AIC'};
table2latex(T,'IC1.tex');
display(T,'Alternative Lag-Order Selection Criteria for VAR Models');
F=table(sichat,hqchat,aichat);
F.Properties.VariableNames = {'SIC' 'HQC' 'AIC'};
display(T,'Lags for maximum Criteria');
table2latex(F,'IC2.tex');
end