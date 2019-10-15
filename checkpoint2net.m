disp("Loading data...")
load("data/trainData", "mfccImgs", "labels");

disp("Loading checkpoint...")
load("checkpoints/net_checkpoint__282__2019_10_15__08_00_34.mat");

m = length(mfccImgs);
P = 0.7;
idx = randperm(m);
trainX = mfccImgs(idx(1:round(P*m)),:); 
validateX = mfccImgs(idx(round(P*m)+1:end),:);
trainY = labels(idx(1:round(P*m)),:); 
validateY = labels(idx(round(P*m)+1:end),:);

trainM = length(trainX);
validateM = length(validateX);

trainX = cell2mat(trainX);
trainX = permute(trainX, [3, 2, 1]);
trainX = reshape(trainX, [3, 299, 299, trainM]);
trainX = permute(trainX, [4, 3, 2, 1]);

validateX = cell2mat(validateX);
validateX = permute(validateX, [3, 2, 1]);
validateX = reshape(validateX, [3, 299, 299, validateM]);
validateX = permute(validateX, [4, 3, 2, 1]);

trainX = permute(trainX, [2, 3, 4, 1]);
validateX = permute(validateX, [2, 3, 4, 1]);

options = trainingOptions("adam", ...
    "MaxEpochs",1, ...
    "MiniBatchSize",32, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",2);

disp("Training network...")
net = trainNetwork(trainX,trainY,layerGraph(net),options);

save("models/net", "net");
disp("Done!")