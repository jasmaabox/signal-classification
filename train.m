%% Create datastore
adsTrain = audioDatastore(fullfile("data/train"), ...
    "IncludeSubfolders", true, ...
    "LabelSource", "foldernames", ...
    "FileExtensions", ".wav");

disp("Reading data...")

inputSize = 65536;
X = zeros(length(adsTrain.Files), inputSize);
Y = zeros(length(adsTrain.Files), 1);

for i=1:length(adsTrain.Files)
    fname = char(adsTrain.Files(i));
    [data, fs] = audioread(fname);
    dataFFT = fft(data, inputSize);
    input = transpose(abs(dataFFT(:,1)));
    
    X(i,:) = input;
    Y(i,:) = adsTrain.Labels(i);
end

%% Fit to multiclass ECOC
disp("Start training...")
model = fitcecoc(X, Y);
compactModel = compact(model);
disp("Done!")

disp("Saving model...")
save("models/classifier", "compactModel")