% Single file testing

fname = input("Enter file path as string: ");

disp("Extracting data...")
T = tall({fname});
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

disp("===")
disp("Class: ")
disp(predY)
disp("Accuracy: ")
disp(max(scores))