function x = path2signal(name)
    [audioIn,Fs] = audioread(char(name));
    [P,Q] = rat(48000/Fs);                    % Resample to 48000
    x = resample(audioIn, P, Q);
    x = mean(x, 2);                           % Convert to mono
    x = x/sqrt(sum(abs(x .^2)) / length(x));  % RMS normalize
end