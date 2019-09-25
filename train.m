% Read data
ds = datastore('test.csv', ...
               'TreatAsMissing','NA');
T = readall(ds)

inputSize = 65536;
X = zeros(length(T.File), inputSize);
Y = zeros(length(T.File), 1);
for i=1:length(T.File)
    fname = char(T.File(i));
    [data, fs] = audioread(fname);
    dataFFT = fft(data, inputSize);
    input = transpose(abs(dataFFT(:,1)));
    
    X(i,:) = input;
    Y(i,:) = T.Class(i);
end

% Fit to multiclass ECOC
model = fitcecoc(X, Y);

% Predict
predict(model, X(1,:))
