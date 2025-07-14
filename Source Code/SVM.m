% === Load cleaned features ===
load('fma_cleaned_features.mat');  % featureList, genreLabels

% === Define consistent target genres ===
targetGenres = ["Folk", "Hip-Hop", "Instrumental", "International"];

% === Filter dataset for selected genres ===
genreCats = categorical(genreLabels);
keepIndices = ismember(genreCats, targetGenres);
filteredFeatures = featureList(keepIndices, :);
filteredLabels = genreCats(keepIndices);

% === Cast to consistent categories ===
filteredLabels = categorical(string(filteredLabels), targetGenres);

% === Manual stratified train-test split (95/5 per class) ===
trainIdx = false(size(filteredLabels));
for i = 1:numel(targetGenres)
    idx = find(filteredLabels == targetGenres(i));
    N = numel(idx);
    nTrain = round(0.9 * N);
    shuffledIdx = idx(randperm(N));
    trainIdx(shuffledIdx(1:nTrain)) = true;
end

% === Train/Test split ===
XTrain = filteredFeatures(trainIdx, :);
YTrain = filteredLabels(trainIdx);
XTest  = filteredFeatures(~trainIdx, :);
YTest  = filteredLabels(~trainIdx);

% === Train SVM model ===
SVMModel = fitcecoc(XTrain, YTrain, ...
    'Coding', 'onevsall', ...
    'Learners', templateSVM('KernelFunction', 'rbf', 'Standardize', true), ...
    'ClassNames', targetGenres);

% === Predict ===
YPred_svm = predict(SVMModel, XTest);

% === Recast predictions and ground truth to consistent categories ===
YTest = categorical(string(YTest), targetGenres);
YPred_svm = categorical(string(YPred_svm), targetGenres);

% === Validate size ===
assert(numel(YPred_svm) == numel(YTest), 'Mismatch in prediction and true label count');

% === Accuracy ===
accuracy = mean(YPred_svm == YTest);
fprintf('🎯 SVM Accuracy (4 Genres): %.2f%%\n', accuracy * 100);

% === Save model and predictions ===
YTrue_svm = YTest;  % Store for consistency in external evaluation
save('trainedSVMModel_4Genres.mat', 'SVMModel', 'YPred_svm', 'YTrue_svm');
