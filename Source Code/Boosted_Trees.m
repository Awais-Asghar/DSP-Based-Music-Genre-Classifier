% === Load preprocessed feature data ===
load('fma_cleaned_features.mat');  % featureList, genreLabels

% === Define target genres ===
targetGenres = ["Folk", "Hip-Hop", "Instrumental", "International"];

% === Filter data for target genres only ===
genreCats = categorical(genreLabels);
keepIdx = ismember(genreCats, targetGenres);

X = featureList(keepIdx, :);
Y = genreCats(keepIdx);

% === Ensure categorical consistency ===
Y = categorical(string(Y), targetGenres);  % enforce consistent categories

% === Train-test split ===
N = numel(Y);
idx = randperm(N);
split = round(0.95 * N);
XTrain = X(idx(1:split), :);
YTrain = Y(idx(1:split));
XTest  = X(idx(split+1:end), :);
YTest  = Y(idx(split+1:end));

% === Train Boosted Tree ===
boostedTreeModel = fitcensemble(XTrain, YTrain, ...
    'Method', 'AdaBoostM2', ...
    'NumLearningCycles', 300, ...
    'Learners', templateTree('MaxNumSplits', 10));

% === Evaluate ===
YPred_tree = predict(boostedTreeModel, XTest);

% Recast predictions and true labels with the same category set
YTest = categorical(string(YTest), targetGenres);
YPred_tree = categorical(string(YPred_tree), targetGenres);

% === Sanity check: Ensure size match ===
assert(numel(YPred_tree) == numel(YTest), 'Mismatch in prediction and ground truth size.');

% === Accuracy ===
acc = mean(YPred_tree == YTest);
fprintf('🎯 Boosted Tree Accuracy: %.2f%%\n', acc * 100);

% === Save model and evaluation data ===
YTest_tree=YTest;  % Save for external use
save('trainedBoostedTreeModel.mat', 'boostedTreeModel', 'YPred_tree', 'YTest_tree');
