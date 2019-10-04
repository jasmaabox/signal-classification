function img = extractMFCC(x)
    x = mean(x, 2); % Collapse stereo?
    x = x/sqrt(sum(abs(x .^2)) / length(x)); % RMS normalize
    mfcc = melSpectrogram(x, 48000);
    img = uint8(floor(normalize(mfcc) * 255)); % Normalize and convert to image
    img = cat(3, img, img, img);
    img = imresize(img,[227 227]);
end