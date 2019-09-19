% Read data
ds = datastore('dummy.csv', ...
               'TreatAsMissing','NA');
T = readall(ds)

for i=1:length(T.File)
    fname = char(T.File(i));
    [data, fs] = audioread(fname);
    dataFFT = fft(data, 2^18);
    input = abs(dataFFT(:,1));
end

inputs = [input1 input2];

%{
% NN
layers = [
    fullyConnectedLayer(512,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(128,"Name","fc_2")
    reluLayer("Name","relu_2")
    fullyConnectedLayer(64,"Name","fc_3")
    reluLayer("Name","relu_3")
    fullyConnectedLayer(10,"Name","fc_4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',inputs, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%}