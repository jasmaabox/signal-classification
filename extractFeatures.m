% Extract feature vector
function features = extractFeatures(x, aFE)
    x = mean(x, 2); % Collapse stereo?
    features = extract(aFE, x);
    features = transpose(features);
    features = fillmissing(features, 'constant',0); % Get rid of NaNs
end