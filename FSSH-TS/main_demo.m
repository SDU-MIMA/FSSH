%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is the code for paper 'fast scalable supervised hashing'.
% This is the code for variant FSSH_ts.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function main_demo()

nbits_set = [16,32,64,96];

%% Load dataset
fprintf('Loading Data...\n');
load('cifar10.mat')

exp_data.traingnd=train_label_cifar;
exp_data.testgnd=test_label_cifar;
cateTrainTest = bsxfun(@eq, train_label_cifar, test_label_cifar');
exp_data.WTT=cateTrainTest';
exp_data.traindata = double(train_data_cifar');
exp_data.testdata = double(test_data_cifar');

for ii=1:length(nbits_set)
    
    nbits=nbits_set(ii);
    
    % FSSH_ts
    MAP =train_FSSH(exp_data, nbits);
    fprintf('MAP result of FSSH: %d...   \n', MAP );
    
    
end
end

