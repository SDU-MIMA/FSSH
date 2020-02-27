function Main_FSSH_CMH()
nbits_set=[16, 32, 64,128];

%% prepare data
fprintf('preparing data...\n')
load('MIRFLICKR.mat');
R = randperm(size(LAll,1));
queryInds = R(1:2000);
sampleInds = R(2001:16738);
XTrain = XAll(sampleInds, :); YTrain = YAll(sampleInds, :); LTrain = LAll(sampleInds, :);
XTest = XAll(queryInds, :); YTest = YAll(queryInds, :); LTest = LAll(queryInds, :);
clear XAll YAll LAll

if isvector(LTrain)
    LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
    LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
end

%% parameters
fprintf('setting parameters...\n')
param.alpha = 500;
param.beta = 100;
param.gamma1 = 100;
param.gamma2 = 100;
param.iter = 3;
param.top_K = 2000;

%% 
for bit=1:length(nbits_set)
    nbits=nbits_set(bit);
    param.nbits=nbits;
    eva_info = evaluate_FSSH_CMH(XTrain,YTrain,XTest,YTest,LTest,LTrain,param);
    
    % train time
    trainT = eva_info.trainT;
    
    % MAP
    Image_to_Text_MAP = eva_info.Image_VS_Text_MAP;
    Text_to_Image_MAP=eva_info.Text_VS_Image_MAP;
    
    fprintf('FSSH-CMH %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',nbits,Image_to_Text_MAP,Text_to_Image_MAP,trainT);
    
end

end
