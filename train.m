% Create datastore
adsTrain = audioDatastore(fullfile("data/dummyTrain"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames", ...
    "FileExtensions", ".wav");
labels = adsTrain.Labels;

% Extract data
disp("Extracting data...")
T = tall(adsTrain);
mfccImgsTall = cellfun( @(x)extractMFCC(x),T, "UniformOutput",false);
mfccImgs = gather(mfccImgsTall);

% Save training data
disp("Saving data...")
save("data/trainData", "mfccImgs", "labels");
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
trainX = reshape(cell2mat(trainX), [trainM, 299, 299, 3]);
validateX = reshape(cell2mat(validateX), [validateM, 299, 299, 3]);
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
    "MaxEpochs",100, ...
    "MiniBatchSize",32, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",2, ...
    'ValidationData',{validateX,validateY}, ...
    'ValidationFrequency',5, ...
    'CheckpointPath','checkpoints');

disp("Training network...")
net = trainNetwork(trainX, trainY,lgraph,options);
save("models/net", "net");