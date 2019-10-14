% Create datastore
ds = datastore('dummyTest.csv', ...
               'TreatAsMissing','NA');

testTable = readall(ds);
adsTest = testTable.Files;
testY = categorical(testTable.Labels);

% Extract features
disp("Extracting data...")
T = tall(adsTest);
audioArr = cellfun( @(x)path2signal(x),T, "UniformOutput",false);
mfccImgsTall = cellfun( @(x)signal2MFCC(x),audioArr, "UniformOutput",false);
mfccImgs = gather(mfccImgsTall);

m = length(mfccImgs);

testX = mfccImgs;
testX = cell2mat(testX);
testX = permute(testX, [3, 2, 1]);
testX = reshape(testX, [3, 299, 299, m]);
testX = permute(testX, [4, 3, 2, 1]);

testX = permute(testX, [2, 3, 4, 1]);

% Load in model
disp("Loading model...")
load("models/net");

% Classify
[predY,scores] = classify(net, testX);

plotconfusion(testY, predY)
set(findobj(gca,'type','text'),'fontsize',10) 