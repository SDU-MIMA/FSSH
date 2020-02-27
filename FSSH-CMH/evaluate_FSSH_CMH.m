function evaluation_info=evaluate_FSSH_CMH(XTrain,YTrain,XTest,YTest,LTest,LTrain,FSSHparam)

%% Training 
[Wx, Wy, B, traintime] = train_FSSH_CMH(XTrain, YTrain, FSSHparam, LTrain);
evaluation_info.trainT=traintime;
fprintf('evaluating...\n')

%% image as query to retrieve text database
BxTest = compactbit(XTest*Wx >= 0);
ByTrain = compactbit(B >= 0);
DHamm = hammingDist(BxTest, ByTrain);
[~, orderH] = sort(DHamm, 2);
evaluation_info.Image_VS_Text_MAP  = mAP(orderH', LTrain, LTest);

%% text as query to retrieve image database
ByTest = compactbit(YTest*Wy >= 0);
BxTrain = compactbit(B >= 0);
DHamm = hammingDist(ByTest, BxTrain);
[~, orderH] = sort(DHamm, 2);
evaluation_info.Text_VS_Image_MAP = mAP(orderH', LTrain, LTest);

end



