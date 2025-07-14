% === Load cleaned features ===
load('fma_cleaned_features.mat');  % featureList, genreLabels

% Convert to categorical
genreCats = categorical(genreLabels);

% === Filter for only the 4 target genres ===
targetGenres = ["Folk", "Hip-Hop","Instrumental","International"];
keepIndices = ismember(genreCats, targetGenres);

% Apply filtering
filteredFeatures = featureList(keepIndices, :);
filteredLabels = genreCats(keepIndices);

% Display unique genres after filtering (sanity check)
disp("Filtered genres:");
disp(unique(filteredLabels));

% === Stratified train/test split (80/20 per genre) ===
uniqueLabels = categories(filteredLabels);
trainIdx = false(size(filteredLabels));

for i = 1:numel(uniqueLabels)
    idx = find(filteredLabels == uniqueLabels(i));
    N = numel(idx);
    nTrain = round(0.9 * N);
    shuffledIdx = idx(randperm(N));
    trainIdx(shuffledIdx(1:nTrain)) = true;
end

XTrain = filteredFeatures(trainIdx, :);
YTrain = filteredLabels(trainIdx,:);
XTest  = filteredFeatures(~trainIdx, :);
YTest  = filteredLabels(~trainIdx,:);

% === Evaluate for different k values ===
kValues = 1:2:21;  % Odd values to avoid tie-breaking
accuracies = zeros(size(kValues));

for i = 1:length(kValues)
    k = kValues(i);
    knnModel = fitcknn(XTrain, YTrain, ...
        'NumNeighbors', k, ...
        'Standardize', true, ...
        'Distance', 'seuclidean');
    
    YPred_knn = predict(knnModel, XTest);
    accuracies(i) = sum(YPred_knn == YTest) / numel(YTest);
    YTrue=YTest;
    fprintf('k = %d → Accuracy = %.2f%%\n', k, accuracies(i) * 100);
end

% === Plot Accuracy vs. k ===
figure;
plot(kValues, accuracies * 100, '-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('k (Number of Neighbors)');
ylabel('Accuracy (%)');
title('k-NN Accuracy vs. k for 4 Genres');
grid on;

% === Save Best Model ===
[~, bestIdx] = max(accuracies);
bestK = kValues(bestIdx);
fprintf('\n✅ Best k = %d with Accuracy = %.2f%%\n', bestK, accuracies(bestIdx) * 100);

% Retrain best model
bestKNNModel = fitcknn(XTrain, YTrain, ...
    'NumNeighbors', bestK, ...
    'Standardize', true, ...
    'Distance', 'euclidean');
save('bestKNNModel_4Genres.mat', 'bestKNNModel','YPred_knn',"YTrue");
