function [VC]=fevd(A,B0inv,K,p,h);
% Structural forecast error variance Decomposition
% using B0inv for a K dimensional VAR and horizon h 
% Input:
% A: companion matrix
% B0inv: Identifying matrix
% K: number of variables
% p: number of lags
% h: horizon 
%
% Output: rows = horizon; first K cols = effect of first shock
%         on all variables
%
% Michael Bergman December 2018
%
J=[eye(K,K) zeros(K,K*(p-1))];
% Compute Theta_1
Theta=J*A^0*J'*B0inv;
% Theta is the structural impulse horizon 1
% take transpose such that rows=variables and cols=shocks
Theta=Theta';
% Compute mse(1)
Theta2=(Theta.*Theta);
Theta_sum=sum(Theta2);
VC=zeros(K,K);
for j=1:K
    VC(j,:)=Theta2(j,:)./Theta_sum;
end;
% Reshape
VC = reshape(VC', [1, K*K]);
% Then compute FEVD for next horizons
% 
Theta_h=Theta2;
for i=2:h
    Theta=J*A^(i-1)*J'*B0inv;
    Theta=Theta';
    Theta2=(Theta.*Theta);
    Theta_h = Theta_h+Theta2;
    Theta_sum=sum(Theta_h);
    for j=1:K
       VChelp(j,:)=Theta_h(j,:)./Theta_sum;
    end;
    VC = [ VC; reshape(VChelp', [1,K*K] )];
end;

end
