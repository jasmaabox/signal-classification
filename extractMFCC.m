function img = extractMFCC(x)
    x = mean(x, 2); % Collapse stereo?
    x = x/sqrt(sum(abs(x .^2)) / length(x)); % RMS normalize
    mfcc = melSpectrogram(x, 48000);
    % convert to image
    img = normalize(mfcc);
    img = uint8(floor(img * 255));
    img = ind2rgb(img, jet);
    img = imresize(img,[227 227]);
end