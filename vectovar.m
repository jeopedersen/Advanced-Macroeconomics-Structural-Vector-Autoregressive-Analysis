function [B]=vectovar(Gamma,Pi);
% This function converts VEC estimates into companion matrix for VAR in
% levels
%
% Michael Bergman
% Checked October 2018
%
% Input: KxKp Gamma matrix
%        KxK Pi = alpha*beta' matrix
% Output: KpxKp Companion matrix B
[K,Kp] = size(Gamma);
p = Kp/K;
B = eye(K) + Pi + Gamma(:,1:K);
Gamma = [ Gamma zeros(K,K) ];
i=1;
j=K;
while j<=Kp
 B = [ B (Gamma(:,i+K:j+K)-Gamma(:,i:j)) ];
 i = i+K;
 j=j+K;
end
B = [B; eye((p)*K) zeros(K*(p),K)];
end




