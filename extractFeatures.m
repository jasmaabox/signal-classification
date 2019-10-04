% Extract feature vector
function features = extractFeatures(x, aFE)
    x = mean(x, 2); % Collapse stereo?
    x = x/sqrt(sum(abs(x .^2)) / length(x)); % RMS normalize
    features
    features = fillmissing(features, 'constant',0); % Get rid of NaNs
end