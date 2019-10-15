function img = signal2MFCC(x)
    mfcc = melSpectrogram(x, 48000);
    img = normalize(mfcc);
    img = uint8(floor(img * 255));
    img = ind2rgb(img, jet);
    img = imresize(img,[299 299]);
end