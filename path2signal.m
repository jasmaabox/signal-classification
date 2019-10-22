% Convert path to signal
function x = path2signal(name)
    [audioIn,Fs] = audioread(char(name));
    [P,Q] = rat(48000/Fs);                    % Resample to 48000
    x = resample(audioIn, P, Q);
    x = mean(x, 2);                           % Convert to mono
    x = lowpass(x, 150, 48000);               % Lowpass filter
    %x = x * sqrt(length(x) / sum(x .^2));  % RMS normalize
    x = x / max(abs(x));                      % Peak normalize
end