% Create datastore
ds = datastore('dummyTest.csv', ...
               'TreatAsMissing','NA');

testTable = readall(ds);
adsTest = testTable.Files;
testY = categorical(testTable.Labels);

% Extract features
disp("Extracting data...")
T = tall(adsTest);
audioArr = cellfun( @(x)path2audio(x),T, "UniformOutput",false);
mfccImgsTall = cellfun( @(x)extractMFCC(x),audioArr, "UniformOutput",false);
mfccImgs = gather(mfccImgsTall);

m = length(mfccImgs);
testX = reshape(cell2mat(mfccImgs), [m, 299, 299, 3]);
testX = permute(testX, [2, 3, 4, 1]);

% Load in LSTM
disp("Loading model...")
load("models/net");

% Classify
[predY,scores] = classify(net, testX);

plotconfusion(testY, predY)


% Convert path to audio
function audioIn = path2audio(x)
    audioIn = audioread(char(x));
end