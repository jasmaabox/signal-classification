load("data/trainData", "featureVectors", "labels");

for i=1:size(featureVectors, 1)
    mat = cell2mat(featureVectors(i));
    if size(find(isnan(mat)), 1) > 0
        i
    end
end

disp("Done!")