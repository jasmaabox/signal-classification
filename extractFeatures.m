% Extract feature vector
function features = extractFeatures(x, aFE)
    x = mean(x, 2); % Collapse stereo?
    features = extract(aFE, x);
    features = transpose(features);
end