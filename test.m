%% Create datastore
% Read data
ds = datastore('test.csv', ...
               'TreatAsMissing','NA');

adsTest = readall(ds);
                
disp("Reading data...")

inputSize = 65536;
X = zeros(length(adsTest.Files), inputSize);
Y = zeros(length(adsTest.Files), 1);
for i=1:length(adsTest.Files)
    fname = char(adsTest.Files(i));
    [data, fs] = audioread(fname);
    dataFFT = fft(data, inputSize);
    input = transpose(abs(dataFFT(:,1)));
    
    X(i,:) = input;
    Y(i,:) = adsTest.Labels(i);
end

%% Load ecoc
disp("Loading model...")
load("models/classifier.mat")

%% Predict and count
predY = predict(compactModel, X);

correct = sum(Y==predY);    % Fancy logical arrays

disp("===")
disp("Number correct:")
disp(correct)