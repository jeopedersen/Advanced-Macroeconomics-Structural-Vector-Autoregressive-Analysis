function [alpha,Gamma,SigmaML,u]=VECMknown(y,p,r,beta);
% This function estimates VEC for known cointegration vector
% and a constant term using ML.
% See KL section 3.2.2.
% Input:
% y=TxK times series vector
% p=number of lags in underlying VAR model
% r=cointegration rank
% beta=Kxr cointegration vector
%
% Output:
% alpha=Kxr adjustment coefficients
% Gamma=Coefficients associated with first difference terms. Kx(KP+1) matrix
% with constants in the first column.
% SigmaML=Variance-covariance matrix of residuals
% u=residuals Kx(T-p)
%
% Michael Bergman December 2018
%
[t,q]=size(y);   % determine size of time series vector
ydif=dif(y);     % take first differences
y=y';
ydif=ydif';
dy=ydif(:,p:t-1);	
X=ones(1,t-p);
for i=1:p-1
 	X=[X; ydif(:,p-i:t-1-i)];
end;

y=y(:,p:t-1);

R0t=dy-(dy*X'/(X*X'))*X;    % Equation relates to (3.2.8)
R1t=y-(y*X'/(X*X'))*X;
% Compute S00, S01 and S11 needed to compute alpha in (3.2.10)
S00=R0t*R0t'/(t-p);
S11=R1t*R1t'/(t-p);
S01=R0t*R1t'/(t-p);

% Compute ML estimates
%
alpha=S01*beta*inv(beta'*S11*beta);   % KL (3.2.10)
Gamma=(dy-alpha*beta'*y)*X'/(X*X');   % KL (3.2.7)
u=dy-alpha*beta'*y-Gamma*X;
SigmaML=u*u'/(t-p);
end

function [ydif]=dif(y)

t=size(y,1);
ydif=y(2:t,:)-y(1:t-1,:);
end