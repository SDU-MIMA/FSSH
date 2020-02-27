function [W1, W2, B, traintime] = train_FSSH_CMH(X, Y, param, L)

%% set the parameters
nbits = param.nbits;
alpha = param.alpha;
beta = param.beta;
gamma1 = param.gamma1;
gamma2 = param.gamma2;

%% 
[n, dX] = size(X);
c = size(L,2);

%% pre-computing
D=L'*L;
C1=pinv(X'*X)*X';
C2=pinv(Y'*Y)*Y';
S=L*L'; S=S>0;S=2*S-1;  E=S*L;

%% initialization
fprintf('training...\n')
tic;

fprintf('initializing...\n')
Z = randn(n, nbits);
G = randn( c, nbits);

%% optimization
for iter = 1:param.iter
    % update B
    B = sign(alpha*L*G+beta*Z);
    
    % update G
    G = pinv(D)*(E'*Z+alpha*L'*B)/(Z'*Z+alpha*eye(nbits));
    
    % update W1
    W1=C1*Z;
    
    % update W2
    W2=C2*Z;
    
    % update Z
    Z=(E*G+beta*B+gamma1*X*W1+gamma2*Y*W2)/(G'*D*G+(beta+gamma1+gamma2)*eye(nbits));
end

fprintf('training is done...\n')
traintime=toc;
end
