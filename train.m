% Create datastore
ds = audioDatastore(fullfile("data/train"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

adsTrain = ds.Files;
labels = ds.Labels;

% Extract data
disp("Extracting data...")
T = tall(adsTrain);
audioArr = cellfun( @(x)path2signal(x),T, "UniformOutput",false);
mfccImgsTall = cellfun( @(x)signal2MFCC(x),audioArr, "UniformOutput",false);
mfccImgs = gather(mfccImgsTall);

% Save training data
disp("Saving data...")
save("data/trainData", "mfccImgs", "labels", '-v7.3');
load("data/trainData", "mfccImgs", "labels");

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

% Transfer learn on inception resnet v2
disp("Loading resnet...")
numClasses = numel(unique(labels));
net = inceptionresnetv2;
lgraph = layerGraph(net);
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'predictions',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);

options = trainingOptions("adam", ...
    "MaxEpochs",10, ...
    "MiniBatchSize",32, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",2, ...
    "ValidationData",{validateX,validateY}, ...
    "ValidationFrequency",5, ...
    "CheckpointPath","checkpoints");

disp("Training network...")
net = trainNetwork(trainX, trainY,lgraph,options);
save("models/net", "net");
disp("Done!")