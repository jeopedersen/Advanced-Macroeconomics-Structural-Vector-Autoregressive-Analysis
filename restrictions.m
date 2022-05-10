% restrictions.m
function q=restrictions(B0inv)
global GAMMA SIGMA alpha beta alpha_perp beta_perp Xi p so
K=size(B0inv,1);
THETA1=Xi*B0inv;
% This is Upsilon 
F=vec(B0inv*B0inv'-so(1:K,1:K));
% Long run and short run restrictions
q=[F; B0inv(1,3); THETA1(1,2); THETA1(1,3); THETA1(1,4); THETA1(2,3); THETA1(2,4);THETA1(3,3); THETA1(3,4); THETA1(4,3); THETA1(4,4)];
q'+1;