% Create datastore
adsTrain = audioDatastore(fullfile("data/train"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames", ...
    "FileExtensions", ".wav");
trainY = adsTrain.Labels;

% Extract data
disp("Extracting data...")
run("makeExtractor.m");
T = tall(adsTrain);
featureVectorsTall = cellfun( @(x)extractFeatures(x,extractor),T, "UniformOutput",false);
featureVectors = gather(featureVectorsTall);

% Save training data
disp("Saving data...")
save("data/trainData", "featureVectors", "trainY");
%load("data/trainData", "featureVectors", "trainY");

% Define and train LSTM
numClasses = numel(unique(trainY));
layers = [ ...
    sequenceInputLayer(45)
    bilstmLayer(50,"OutputMode","sequence")
    dropoutLayer(0.1)
    bilstmLayer(50,"OutputMode","last")
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

miniBatchSize = 128;
options = trainingOptions("adam", ...
    "MaxEpochs",100, ...
    "MiniBatchSize",miniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "Shuffle","every-epoch", ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.1, ...
    "LearnRateDropPeriod",2);

disp("Training network...")
net = trainNetwork(featureVectors,trainY,layers,options);
save("models/net", "net");