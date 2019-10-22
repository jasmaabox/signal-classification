% Create datastore
ds = audioDatastore(fullfile("data/dummyTrain"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames");

% Extract data
disp("Extracting data...")
signal = path2signal(ds.Files(42));
%plot(signal)
mfccImg = signal2MFCC(signal);
image(mfccImg)