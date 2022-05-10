%% Loading data
clear;
clearvars;
clc;
set(0,'DefaultLegendAutoUpdate','off');
load('11.mat');
global  beta Xi p K r Gamma so
%% Defining the variables
y_t = Y(:,1); % Gross Domestic Product (GDP) - log
p_t = Y(:,2); % Price level - log
i_t = Y(:,3); % Interest rate (the policy rate)
e_t = Y(:,4); % Exchange rate (measured as home currency units per unit foreign currency) - log


%% The Data
%  Problem 1
figure
subplot(2,2,1);
plot(dates,y_t);
title('Gross Domestic Product (GDP)');
axis tight

subplot(2,2,2);
plot(dates,p_t);
title('Price Level (P)');
axis tight

subplot(2,2,3);
plot(dates,i_t);
title('Policy Rate (I)');
axis tight

subplot(2,2,4);
plot(dates,e_t);
title('Exchange Rate (E)');
axis tight
%% Lag Order Determination and Diagnostic Testing
%% Problem 5
x_t = [y_t p_t i_t e_t];
% KL chooses pmax
pmax=12*((length(x_t)/100)^(1/4));
pmax=round(pmax,0);
% Specifying number of lags with information criteria
information_criteria = pfinduniv(x_t,pmax);
[sichat,hqchat,aichat]=pfinduniv(x_t,pmax);
p_sichat = find(sichat==min(sichat(:,1)))-1;
p_hqchat = hqchat;
p_aichat = aichat;
% Specifying the number of lags with top-down sequential testing
topdown(x_t,pmax);
%% Problem 6
p = p_aichat;
[Beta,CovBeta,tratioBeta,res,indep,so] = VARls(x_t,p,1,0);
eigenvalues = stabVAR(comp(Beta,p));
maxeigen=max(eigenvalues);
% Testing for multivariate autocorrelation
h = pmax;
[Q_h, pValue, mQ_h, mpValue] = Portmanteau(res,h,p);
% Testing for multivariate heteroskedasticity
march(res,p,width(x_t));
% Testing for multivariate non-normality
multnorm(res);
% Not necessary - test for univariate autocorrelation and ARCH
% Testing for univariate autocorrelation
[Q, pValueLBuni] = LBuni(res,p)
[~,pValue,~,~] = lbqtest(res(:,1),'lags',[p]);
[~,pValue,~,~] = lbqtest(res(:,2),'lags',[p]);
[~,pValue,~,~] = lbqtest(res(:,3),'lags',[p]);
[~,pValue,~,~] = lbqtest(res(:,4),'lags',[p]);
% Testing for univariate heteroskedasticity
[Q, pValuesUARCH] = uARCH(res,p)
[~,pValue,~,~] = archtest(res(:,1),'lags',[p]);
[~,pValue,~,~] = archtest(res(:,2),'lags',[p]);
[~,pValue,~,~] = archtest(res(:,3),'lags',[p]);
[~,pValue,~,~] = archtest(res(:,4),'lags',[p]);
%% Testing for Co-integration
%% Problem 7
% We note that VECM(p-1) is equal to VAR(p)
pp = p-1;
% Testing for co-integration using the Johansen method
% H1 = There are intercepts in the cointegrated series and there are deterministic linear trends in the levels of
% the data.
[hH1,pValueH1,statH1,cValueH1,mlesH1] = jcitest(x_t,'model','H1','lags',pp); 

% H* = There are intercepts and linear trends in the cointegrated series and there are deterministic
% linear trends in the levels of the data.
[hH_star,pValueH_star,statH_star,cValueH_star,mlesH_star] = jcitest(x_t,'model','H*','lags',pp);

 % Sensitivy analysis
 SA = [];
 v = [];
 for pp1 = 1:pmax
      t = table2array(jcitest(x_t,'model','H*','lags',pp1)); 
      v = [v;pp1];
     SA = [SA;sum(t)];
 end   
 SA_t = array2table(SA);
 SA_t1 = array2table(v);
 SA_t2 = [SA_t1,SA_t];
 table2latex(SA_t2,'SA.tex');
%% Problem 8
% Testing the null hypothsis of no linear tren ||d in the co-integration
% vector
r = 2; % rank that we found earlier
[h0,pValue0,stat0,cValue0,mles0] = jcontest(x_t,r,'Bcon',{[0 0 0 0 1]'},'lags',pp,'model','H*');
display('Testing trend stationarity');
display(mles0.paramVals.B,'Estimated beta under restriction');
display(mles0.paramVals.d0,'Estimated trend under restriction');
display([ stat0 pValue0],'LR-test pvalue');
% We reject the null that trend = 0.
%% Problem 9
% Testing stationarity, exclusion and weak exogeneity
SEWE(x_t,pp,r,'H1')
%% Problem 10
format short
ebeta_normalised_3 = mlesH1.r2.paramVals.B(:,1)'./mlesH1.r2.paramVals.B(4,1); 
ebeta_normalised_4 = mlesH1.r2.paramVals.B(:,2)'./mlesH1.r2.paramVals.B(3,2); 
%% Problem 11
% Testing first theoretical co-integration vector
RR1 = [0 -1 0 1]'
[h1,pValue1,stat1,cValue1,mles1] = jcontest(x_t,r,'Bvec',RR1,'lags',pp,'model','H1');
display(mles1.paramVals.B,'Estimated co-integration vector');
display([ stat1 pValue1],'Testing hypotheses on beta');
% Testing second tehoretical co-integration
RR2 = [0 0 1 0]'
[h2,pValue2,stat2,cValue2,mles2] = jcontest(x_t,r,'Bvec',RR2,'lags',pp,'model','H1');
display(mles2.paramVals.B,'Estimated co-integration vector');
display([ stat2 pValue2],'Testing hypotheses on beta');
% Testing together
RR3 = [0 -1 0 1;0 0 1 0]'
[h3,pValue3,stat3,cValue3,mles3] = jcontest(x_t,r,'Bvec',RR3,'lags',pp,'model','H1');
display(mles3.paramVals.B,'Estimated co-integration vector');
display([ stat3 pValue3],'Testing hypotheses on beta');
%% Problem 12
figure
subplot(2,2,1)
plot(dates,ebeta_normalised_3*x_t')
title('1. Estimated Co-integrating Vector (PPP)')
axis tight

subplot(2,2,2)
plot(dates,ebeta_normalised_4*x_t')
title('2. Estimated Co-integrating Vector (Fisher Relation)')
axis tight

subplot(2,2,3)
plot(dates,(RR1'*x_t')')
title('3. Theoretical Co-integrating Vector (PPP)')
axis tight

subplot(2,2,4)
plot(dates,(RR2'*x_t')')
title('4. Theoretical Co-integrating Vector (Fisher Relation)')
axis tight
%% Identification of Structural Model
%% Problem 13
% PPP alone
display(mles1.paramVals.A, '0 -1 0 1')
% PPP both
display(mles3.paramVals.A, '0 -1 0 1; 0 0 1 0')
% Interest rate differential
display(mles2.paramVals.A, '0 0 1 0')
% We find the expected signs for the alpha coefficients w.r.t PPP and
% Fisher 
%% Problem 14
% See Overleaf
%% Impulse Responses and Forecast Error Variance Decomposition
format short
% Problem 15
[t,K] = size(x_t);
q=K;
%beta_r = [ebeta_normalised_3;ebeta_normalised_4]';
beta_r = [0 -1 0 1; 0 0 1 0]';
beta_i = beta_r; % Change
[alpha1,Gamma1,so,res]=VECMknown(x_t,p,r,beta_i);
V = Gamma1(:,1);
Gamma = Gamma1(:,2:(p-1)*K+1);
[A] = vectovar(Gamma,alpha1*beta_i'); 
a = A(1:K,:);
beta=beta_r;

% Use the matlab function null to compute orthogonal complements
beta_perp=null(beta')
alpha_perp=null(alpha1')

% Compute GammaSum
GammaSum=Gamma(1:q,1:q);
if p>2
for i=1:p-2
   GammaSum=GammaSum+Gamma(1:q,i*q+1:i*q+q);
end;
end

% Compute Xi=C(1)
Xi=beta_perp*inv(alpha_perp'*(eye(q)-GammaSum)*beta_perp)*alpha_perp'

% Step 2: Method of Moments
% Set seed (to ensure replicability of fsolve results)
randn('seed',1) 

% Set some options for fsolve
warning off
options=optimset('TolX',1e-10,'TolFun',1e-10,'MaxFunEvals',1e+10);


% Compute B_{0}^{-1}
B0inv=fsolve('restrictions',randn(q),options); 
display('Unnormalized structural impact multiplier matrix');
% Ensuring the diagonal is positive
for j = 1:4
    if B0inv(j,j) < 0
        B0inv(:,j) = -B0inv(:,j);
    end
end
display(B0inv);

% Estimate of Upsilon
Upsilon = Xi*B0inv;
display(Xi*B0inv,'Upsilon');

% Covariance matrix structural shocks, Sigma_w Should be an identity matrix
display(inv(B0inv)*so*inv(B0inv)','Covariance matrix structural shocks');

%% Problem 16

% Estimating IRF's (using the standard residual based recursive design
% bootstrap and IRF confidence bands computed using the Delta method)

tic
% Specify the VEC model
p=p;     % number of lags in underlying VAR
r=r;     % cointegration rank
h=60;    % horizon for IRFs
hFEVD=40;  % horizon for FEVD Note: Not used in this code
con=1;     % constant in VEC
tr=0;      % no linear trend = 0, linear trend=1
y = x_t;
% Prepare Bootstrap dummies
boot = 1;   %=1 bootstrap p consequtive initial conditions, =0 use actual
            %   initial condition.
BS_type = 0;  % BS_type = 0: standard recursive bootstrap. BS_type = 1: Wild bootstrap
show = 0;    % show = 1: display identification conditions for each BS
% Counting VEC models that do not satisfy I(1,r) condition  

fail = 0;


ntrials=500;  % Number of trials. Total number of trials = ntrials+fail
sign=1.96;       % Significance level: 1=68% confidence bands

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code below should work for any VEC model but remember to make adjustments
% to restrictions.m if you use other identifying restrictions and
% add/change the code showing the graph and the FEVDs if other shocks are
% of interest. Comments/reminders are in the code below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute sample size and number of observations
[T,K]=size(x_t);

% Step 1: Estimate VECM for known beta

[alpha,Gamma,SIGMA,res]=VECMknown(x_t,p,r,beta);
u = res;
V=Gamma(:,1);                  % Constant
GAMMA = Gamma(:,2:K*(p-1)+1);   % first diff lags
% Use the matlab function null to compute orthogonal complements
beta_perp=null(beta')
alpha_perp=null(alpha')

% Compute GammaSum
GammaSum=GAMMA(1:K,1:K);
if p>2
for i=1:p-2
   GammaSum=GammaSum+GAMMA(1:K,i*K+1:i*K+K);
end;
end

% Compute Xi=C(1)
Xi=beta_perp*inv(alpha_perp'*(eye(K)-GammaSum)*beta_perp)*alpha_perp'
Xiorg=Xi;
SIGMAorg=SIGMA;
% Compute B0inv
% Set some options for fsolve
warning off
options=optimset('TolX',1e-10,'TolFun',1e-10,'MaxFunEvals',1e+10,'Display','off');
% Use seed to allow for replication
% Initial guess of B0inv
B0invinit = randn(K,K);

% Use solver to compute B_{0}^{-1}
% Remember to change restrictions.m to reflect the identifying restrictions
% you would like to impose!
B0inv=fsolve('restrictions',B0invinit,options);

% We need to switch sign such that we add a positive balanced growth shock
for j = 1:q 
  if B0inv(j,j) <0
  B0inv(:,j)=-B0inv(:,j);
  end
end

% Save the estimated B0inv matrix under new name
% This is just for book-keeping
B0invorg=B0inv;

display('Unnormalized structural impact multiplier matrix');
display(B0inv);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Check that B0inv solution is correct (result should be K x K zero matrix)
display(B0inv*B0inv'-SIGMA,'Check that B0inv is correct, should be a zero matrix');

% Estimate of Upsilon
Upsilon = Xi*B0inv;
display(Xi*B0inv,'Upsilon');

% Covariance matrix structural shocks, Sigma_w Should be I_3
display(inv(B0inv)*SIGMA*inv(B0inv)','Covariance matrix structural shocks');


% Create companion matrix A using function vectovar
% Note that this function computes the companion matrix
% of a VAR in levels from VEC
[A] = vectovar(GAMMA,alpha*beta');
a = A(1:K,:);

% Compute structural impulse responses 
IRF=irfstruc(A,B0inv,K,p,h);
FEVD=fevd_irf(IRF,K,h);

% We have now estimated the VAR using LS and computed IRFs

% Prepare matrices for Bootstrap
Tbig = length(u);     % time periods to simulate

% Define IRF matrix used to compute the standard deviation of bootstrap
% IRFs
IRFmat=zeros(ntrials,(K^2)*(h+1));

% Bootstrap
tottrials = 0;     % Counting total trials

trial=1;
while trial<=ntrials
  display(trial,'This is trial');
  
% Prepare initial conditions
% boot = 0: use actual initial conditions
% boot = 1: use a sequence of p consecutive initial conditions drawn from
% uniform distribution
% This dummy must be defined
if boot==0
  y0p = y(1:p,1:K);
else
  pos=randi([p+1,T]);    % Draw the largest position for initial condition
                         % minimum=p+1 and maximum=t
  y0p = y(pos-p+1:pos,:);  % Define initial conditions starting
                           % with pos and then add previous p-1 values
end

yr = zeros(K,Tbig);   % simulated y for t=1:Tbig equal to zero
yr = [y0p' yr];       % Add initial values

% Draw with replacement from uHat
indexur=randi([1,Tbig],Tbig,1);    % Using randi instruction! Generate Tbig
% random integers from the uniform distribution between 1 and Tbig. Indexur
% is a Tbig x 1 vector with indices.
ur = u(:,indexur);
ur = [ zeros(K,p) ur ];           % Add residuals for initial conditions

% Handle deterministic components: Only constant and linear trend allowed
if con==1;
 determ = ones(Tbig+p,1);
end
if tr==1;
 determ = [ determ (-p+1:length(Tbig))' ];
end


i=p+1;
j=1;
while i<=Tbig+p
  index = flip((j:j+p-1));       % Need to sort yr to correspond to the a matrix
  ylags = vec(yr(:,[ index ]));  % Compute ylags
  if BS_type==0
     % Non-parametric BS  
     yr(:,i)=V*determ(i,:)'+a*ylags+ur(:,i);
  else
     % If Wild Gaussian BS, multiply residuals with random number
     yr(:,i)=V*determ(i,:)'+a*ylags+randn*ur(:,i);  
  end
  i=i+1;
  j=j+1;
end


% Then estimate the VEC model using yr, remember that yr must be transposed
% below!

[alphar,Gammar,SIGMAr,urr]=VECMknown(yr',p,r,beta);

% Check Stability
[AA]=vectovar(Gammar(:,2:K*(p-1)+1),alphar*beta');
[e]=sort(abs(eig(AA)),'descend');
if max(abs(e))>1.000000000001
    fail = fail+1;
    tottrials = tottrials+1;
else
    Vr=Gammar(:,1);                  % Constant
    GAMMAr = Gammar(:,2:K*(p-1)+1);   % first diff lags
% Use the matlab function null to compute orthogonal complements
    beta_perpr=null(beta');
    alpha_perpr=null(alphar');

% Compute GammaSum
    GammaSumr=GAMMAr(1:K,1:K);
    if p>2
    for i=1:p-2
       GammaSumr=GammaSumr+GAMMAr(1:K,i*K+1:i*K+K);
    end
   end

% Compute Xi=C(1)
   Xir=beta_perpr*inv(alpha_perpr'*(eye(K)-GammaSumr)*beta_perpr)*alpha_perpr';

% Compute B0inv
% Set some options for fsolve
   warning off
%options=optimset('TolX',1e-10,'TolFun',1e-10,'MaxFunEvals',1e+10);

% Compute B_{0}^{-1}

   Xi=Xir;
   so=SIGMAr;
   B0inv=fsolve('restrictions',randn(K,K),options);
   for j = 1:q
       if B0inv(j,j) < 0
           B0inv(:,j) = -B0inv(:,j);
       end
   end

   if show==1
   display('Unnormalized structural impact multiplier matrix');
   display(B0inv);
% Check that B0inv solution is correct (result should be K x K zero matrix)
   display(B0inv*B0inv'-SIGMAr,'Check that B0inv is correct, should be a zero matrix');
   end
% Estimate of Upsilon
   Upsilon = Xi*B0inv;
   if show==1
    display(Xi*B0inv,'Upsilon');
   % Covariance matrix structural shocks, Sigma_w Should be I_3
    display(inv(B0inv)*SIGMA*inv(B0inv)','Covariance matrix structural shocks');
   end

% Create companion matrix A using function vectovar
% Note that this function computes the companion matrix
% of a VAR in levels from VEC
   [Ar] = vectovar(GAMMAr,alphar*beta');
% Compute structural impulse responses 
   IRFr=irfstruc(Ar,B0inv,K,p,h);
   IRFmat(trial,:)=vec(IRFr');

   [FEVDr]= fevd_irf(IRFr,K,h); 
   FEVDmat(trial,:)=vec(FEVDr');

   trial = trial+1;
   tottrials = tottrials+1;
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of general code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% The number of total replications, failures and replications
display([tottrials fail ntrials],'Total replications, failures and number of trials');
toc
%%
% Constructing confidence bands using the Delta method
 IRFrstd=reshape((std(IRFmat)'),K^2,h+1); 
 CI1LO=IRF-sign*IRFrstd';
 CI1UP=IRF+sign*IRFrstd';

 % Plotting IRFs with confidence bands
 figure
 subplot(3,2,1)
 plot(0:1:h,IRF(:,9),'r-',0:1:h,CI1UP(:,9),'b:',0:1:h,CI1LO(:,9),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Monetary Policy Shock \rightarrow Gross Domestic Product (GDP)','fontsize',10)
 xlabel('Quarters','fontsize',10)
 subplot(3,2,2)
 plot(0:1:h,IRF(:,10),'r-',0:1:h,CI1UP(:,10),'b:',0:1:h,CI1LO(:,10),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Monetary Policy Shock \rightarrow Price Level','fontsize',10)
 xlabel('Quarters','fontsize',10)
 subplot(3,2,3)
 plot(0:1:h,IRF(:,11),'r-',0:1:h,CI1UP(:,11),'b:',0:1:h,CI1LO(:,11),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Monetary Policy Shock \rightarrow Policy Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 subplot(3,2,4)
 plot(0:1:h,IRF(:,12),'r-',0:1:h,CI1UP(:,12),'b:',0:1:h,CI1LO(:,12),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Monetary Policy Shock \rightarrow Exchange Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
%%
 % Constructing FEVDs using Efron's percentile intervals
CI=prctile(FEVDmat,[2.5]);
CI_1=prctile(FEVDmat,[97.5]);
CI_L=reshape((CI),K^2,h);
CI_H=reshape((CI_1),K^2,h);

% FEVD: Positive permanent foreign shock
 figure
 subplot(3,2,1);
 plot(1:h,FEVD(:,9),'r-',1:h,CI_L(9,:),'b:',1:h,CI_H(9,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Monetary Policy Shock \rightarrow Gross Domestic Product (GDP)')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,2,2);
 plot(1:h,FEVD(:,10),'r-',1:h,CI_L(10,:),'b:',1:h,CI_H(10,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Monetary Policy Shock \rightarrow Price Level')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,2,3);
 plot(1:h,FEVD(:,11),'r-',1:h,CI_L(11,:),'b:',1:h,CI_H(11,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Monetary Policy Shock \rightarrow Policy Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,2,4);
 plot(1:h,FEVD(:,12),'r-',1:h,CI_L(12,:),'b:',1:h,CI_H(12,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Monetary Policy Shock \rightarrow Exchange Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight
%% Problem 17
 figure
 subplot(3,4,1)
 plot(0:1:h,IRF(:,1),'r-',0:1:h,CI1UP(:,1),'b:',0:1:h,CI1LO(:,1),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Productivity Shock \rightarrow Gross Domestic Product (GDP)','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,2)
 plot(0:1:h,IRF(:,2),'r-',0:1:h,CI1UP(:,2),'b:',0:1:h,CI1LO(:,2),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Productivity Shock \rightarrow Price Level','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,3)
 plot(0:1:h,IRF(:,3),'r-',0:1:h,CI1UP(:,3),'b:',0:1:h,CI1LO(:,3),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Productivity Shock \rightarrow Policy Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,4)
 plot(0:1:h,IRF(:,4),'r-',0:1:h,CI1UP(:,4),'b:',0:1:h,CI1LO(:,4),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Productivity Shock \rightarrow Exchange Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,5)
 plot(0:1:h,IRF(:,5),'r-',0:1:h,CI1UP(:,5),'b:',0:1:h,CI1LO(:,5),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Price Shock \rightarrow Gross Domestic Product (GDP)','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,6)
 plot(0:1:h,IRF(:,6),'r-',0:1:h,CI1UP(:,6),'b:',0:1:h,CI1LO(:,6),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Price Shock \rightarrow Price Level','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,7)
 plot(0:1:h,IRF(:,7),'r-',0:1:h,CI1UP(:,7),'b:',0:1:h,CI1LO(:,7),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Price Shock \rightarrow Policy Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,8)
 plot(0:1:h,IRF(:,8),'r-',0:1:h,CI1UP(:,8),'b:',0:1:h,CI1LO(:,8),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Price Shock \rightarrow Exchange Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,9)
 plot(0:1:h,IRF(:,13),'r-',0:1:h,CI1UP(:,13),'b:',0:1:h,CI1LO(:,13),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Exchange Rate Shock \rightarrow Gross Domestic Product (GDP)','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,10)
 plot(0:1:h,IRF(:,14),'r-',0:1:h,CI1UP(:,14),'b:',0:1:h,CI1LO(:,14),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Exchange Rate Shock \rightarrow Price Level','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,11)
 plot(0:1:h,IRF(:,15),'r-',0:1:h,CI1UP(:,15),'b:',0:1:h,CI1LO(:,15),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Exchange Rate Shock\rightarrow Policy Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,12)
 plot(0:1:h,IRF(:,16),'r-',0:1:h,CI1UP(:,16),'b:',0:1:h,CI1LO(:,16),'b:',0:1:h,zeros(1,h+1),'linewidth',1);
 title('Exchange Rate Shock \rightarrow Exchange Rate','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight
%%
 figure
 subplot(3,4,1);
 plot(1:h,FEVD(:,1),'r-',1:h,CI_L(1,:),'b:',1:h,CI_H(1,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Productivity Shock  \rightarrow Gross Domestic Product (GDP)')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,2);
 plot(1:h,FEVD(:,2),'r-',1:h,CI_L(2,:),'b:',1:h,CI_H(2,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Productivity Shock  \rightarrow Price Level')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,3);
 plot(1:h,FEVD(:,3),'r-',1:h,CI_L(3,:),'b:',1:h,CI_H(3,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Productivity Shock \rightarrow Policy Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,4);
 plot(1:h,FEVD(:,4),'r-',1:h,CI_L(4,:),'b:',1:h,CI_H(4,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Productivity Shock \rightarrow Exchange Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,5);
 plot(1:h,FEVD(:,5),'r-',1:h,CI_L(5,:),'b:',1:h,CI_H(5,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Price Shock  \rightarrow Gross Domestic Product (GDP)')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,6);
 plot(1:h,FEVD(:,6),'r-',1:h,CI_L(6,:),'b:',1:h,CI_H(6,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Price Shock  \rightarrow Price Level')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,7);
 plot(1:h,FEVD(:,7),'r-',1:h,CI_L(7,:),'b:',1:h,CI_H(7,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Price Shock \rightarrow Policy Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,8);
 plot(1:h,FEVD(:,8),'r-',1:h,CI_L(8,:),'b:',1:h,CI_H(8,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Price Shock \rightarrow Exchange Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,9);
 plot(1:h,FEVD(:,13),'r-',1:h,CI_L(13,:),'b:',1:h,CI_H(13,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Exchange Rate Shock  \rightarrow Gross Domestic Product (GDP)')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,10);
 plot(1:h,FEVD(:,14),'r-',1:h,CI_L(14,:),'b:',1:h,CI_H(14,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Exchange Rate Shock  \rightarrow Price Level')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,11);
 plot(1:h,FEVD(:,15),'r-',1:h,CI_L(15,:),'b:',1:h,CI_H(15,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Exchange Rate Shock \rightarrow Policy Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight

 subplot(3,4,12);
 plot(1:h,FEVD(:,16),'r-',1:h,CI_L(16,:),'b:',1:h,CI_H(16,:),'b:',1:h,zeros(size(0:h-1)),'k-','linewidth',1)
 title('Exchange Rate Shock \rightarrow Exchange Rate')
 ylabel('Share','fontsize',10)
 xlabel('Quarters','fontsize',10)
 axis tight
%% Problem 18
% Done
%% Problem 20
format long
% Identification of transitory shock
% This process is not automatic
r=2;
p=2;
% Note: cointegration vector is not normalized
beta=[0 -1 0 1; 0 0 1 0]';
K=4;

sigmahat = so;
% Impose restrictions to identify permanent shock
% Upsilon_0

Upsilon0 = [1 0; 0 1; 0 0; 0 1];

% Impose restrictions to identify transitory shocks
% Code below use Umat = TID
% If you want to use an automatic selection,
% set TID=0.

TID = [1 0 1 0; 1 0 0 0];

%------ No need to change code below --------
MHLP=inv(Upsilon0'*Upsilon0)*Upsilon0'*Xi;                   
pipit=MHLP*sigmahat*MHLP';            
pimat=chol(pipit)';                   
Upsilon=Upsilon0*pimat;                       
Fk=inv(Upsilon'*Upsilon)*Upsilon'*Xi;
display(Fk,'Fk matrix');

% Identification transitory shocks

Umat=zeros(r,K);

% If TID=0, use automatic Umat, otherwise use TID defined above
if TID==0;
i=1;
while i<=r;
 Umat(i,K-i+1)=1;
 i=i+1;
end;
else
    Umat = TID;
end

% Check that identification of transitory shocks is valid   
if det(Umat*alpha1)==0
   display('Identification of transitory shock is invalid');
else
   display('Identification of transitory shocks is valid');
end
xi=alpha1*inv(Umat*alpha1);
i=1;
while i<=K;
   j=1;
   while j<=r;
      if abs(xi(i,j))<=1E-12; % just to make sure that elements are = 0
         xi(i,j)=0;
      else
      end
      j=j+1;
   end
   i=i+1;
end

qr=chol(xi'*inv(sigmahat)*xi)';                    
Fr=inv(qr)*xi'*inv(sigmahat);
display(Fr,'Fr matrix');

% Putting it all together to compute B0inv

invB0 = inv([Fk;Fr]);

% Display result and compare to solver solution

display(invB0,'B0^{-1} matrix');
display(B0inv,'Compare to solver');


% Check that identification is valid

display(beta'*Xi,'(1) beta*Xi should be zero');
display(beta'*Upsilon0,'(2) beta*Upsilon_0 should be zero');
display(-Xi*invB0,'(3) C(1)*B0^{-1} should be Upsilon~zeros(K,r)');
display(Upsilon,'where Upsilon');
display(inv(invB0)*so*inv(invB0)','(4) Covariance matrix of structural shocks w_t should be I_K');
display(inv(qr)*xi'*inv(so)*xi*inv(qr'),'Should be diagonal');

