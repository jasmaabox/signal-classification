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
trainX = reshape(cell2mat(trainX), [trainM, 227, 227, 3]);
validateX = reshape(cell2mat(validateX), [validateM, 227, 227, 3]);
trainX = permute(trainX, [2, 3, 4, 1]);
validateX = permute(validateX, [2, 3, 4, 1]);

% Define and train LSTM
numClasses = numel(unique(labels));
layers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([2 2],16,"Name","conv_1","Padding","same")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1","Padding","same")
    dropoutLayer(0.2,"Name","dropout_1")
    convolution2dLayer([2 2],32,"Name","conv_2","Padding","same")
    reluLayer("Name","relu_2")
    maxPooling2dLayer([2 2],"Name","maxpool_2","Padding","same")
    dropoutLayer(0.2,"Name","dropout_2")
    convolution2dLayer([2 2],64,"Name","conv_3","Padding","same")
    reluLayer("Name","relu_3")
    maxPooling2dLayer([2 2],"Name","maxpool_3","Padding","same")
    dropoutLayer(0.2,"Name","dropout_3")
    convolution2dLayer([2 2],128,"Name","conv_4","Padding","same")
    reluLayer("Name","relu_4")
    maxPooling2dLayer([2 2],"Name","maxpool_4","Padding","same")
    dropoutLayer(0.2,"Name","dropout_4")
    globalAveragePooling2dLayer("Name","gapool")
    fullyConnectedLayer(numClasses)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

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
net = trainNetwork(trainX, trainY,layers,options);
save("models/net", "net");