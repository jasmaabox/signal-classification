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
mfccImgsTall = cellfun( @(x)extractMFCC(x),audioArr, "UniformOutput",false);
testX = gather(mfccImgsTall);

m = length(testX);
testX = reshape(cell2mat(testX), [m, 227, 227, 3]);
testX = permute(testX, [2, 3, 4, 1]);

% Load in LSTM
disp("Loading model...")
load("models/net");

% Classify
[predY,scores] = classify(net, testX);

correct = 0;
for i=1:length(testY)
    if testY(i) == predY(i)
        correct = correct + 1;
    end
end

disp("===")
disp("Number correct:")
disp(correct)

disp(scores)


% Convert path to audio
function audioIn = path2audio(x)
    audioIn = audioread(char(x));
end