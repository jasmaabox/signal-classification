% Create datastore
ds = datastore('test.csv', ...
               'TreatAsMissing','NA');

testTable = readall(ds);
adsTest = testTable.Files;
testY = categorical(testTable.Labels);

% Extract features
disp("Extracting data...")
T = tall(adsTest);
audioArr = cellfun( @(x)path2audio(x),T, "UniformOutput",false);
run("makeExtractor.m");
featureVectorsTall = cellfun( @(x)extractFeatures(x, extractor),audioArr, "UniformOutput",false);
featureVectors = gather(featureVectorsTall);

% Load in LSTM
disp("Loading model...")
load("models/net");

% Classify
predY = classify(net, featureVectors);

correct = 0;
for i=1:length(testY)
    if testY(i) == predY(i)
        correct = correct + 1;
    end
end

disp("===")
disp("Number correct:")
disp(correct)



% Convert path to audio
function audioIn = path2audio(x)
    audioIn = audioread(char(x));
end