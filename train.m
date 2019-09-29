% Create datastore
adsTrain = audioDatastore(fullfile("data/dummyTrain"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames", ...
    "FileExtensions", ".wav");
labels = adsTrain.Labels;

% Extract data
disp("Extracting data...")
run("makeExtractor.m");
T = tall(adsTrain);
featureVectorsTall = cellfun( @(x)extractFeatures(x,extractor),T, "UniformOutput",false);
featureVectors = gather(featureVectorsTall);

% Save training data
disp("Saving data...")
save("data/trainData", "featureVectors", "labels");
load("data/trainData", "featureVectors", "labels");

m = size(featureVectors, 1);
P = 0.7;
idx = randperm(m);
trainX = featureVectors(idx(1:round(P*m)),:); 
validateX = featureVectors(idx(round(P*m)+1:end),:);
trainY = labels(idx(1:round(P*m)),:); 
validateY = labels(idx(round(P*m)+1:end),:);

% Define and train LSTM
numClasses = numel(unique(labels));
layers = [ ...
    sequenceInputLayer(91)
    bilstmLayer(50,"OutputMode","sequence")
    dropoutLayer(0.1)
    bilstmLayer(50,"OutputMode","last")
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions("adam", ...
    "MaxEpochs",10, ...
    "MiniBatchSize",32, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",2, ...
    'ValidationData',{validateX,validateY}, ...
    'ValidationFrequency',5);

disp("Training network...")
net = trainNetwork(trainX,trainY,layers,options);
save("models/net", "net");