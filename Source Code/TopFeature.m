% === Load Extracted Feature Data ===
load('fma_advanced_features_max5000.mat', 'featureList', 'genreLabels');

% === Convert Labels to Categorical and Filter Target Genres ===
genreCats = categorical(genreLabels);
targetGenres = ["Folk", "Hip-Hop", "Instrumental", "International"];
keepIdx = ismember(genreCats, targetGenres);

X = featureList(keepIdx, :);
Y = genreCats(keepIdx);

% === Define Feature Names Manually Based on Your Extractor Settings ===
baseNames = {
    'mfcc', 'SpectralCentroid', 'ZeroCrossRate', 'SpectralRolloffPoint', ...
    'SpectralFlux', 'SpectralEntropy', 'SpectralSpread', 'SpectralSkewness', ...
    'SpectralKurtosis', 'SpectralCrest', 'harmonicRatio', 'pitch'
};

% Dimensions of each feature: mfcc = 13, rest = 1
baseDims = [13, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

% Generate full label list
finalFeatureLabels = {};
for i = 1:numel(baseNames)
    if baseDims(i) == 1
        finalFeatureLabels{end+1} = baseNames{i};
    else
        for j = 1:baseDims(i)
            finalFeatureLabels{end+1} = sprintf('%s_%d', baseNames{i}, j);
        end
    end
end

% === Train Boosted Tree Model ===
boostModel = fitcensemble(X, Y, ...
    'Method', 'AdaBoostM2', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 10));

% === Get Feature Importance ===
importance = predictorImportance(boostModel);
finalFeatureLabels = finalFeatureLabels(1:numel(importance));  % Ensure alignment

% === Sort and Plot Top 20 Features ===
[sortedImp, idx] = sort(importance, 'descend');
topN = min(5, length(importance));  % Adjust if fewer than 20

figure('Name', 'Top Audio Features by Importance', 'Color', 'w');
barh(sortedImp(1:topN), 'FaceColor', [0.2 0.4 0.6]);
set(gca, 'YTickLabel', finalFeatureLabels(idx(1:topN)));
set(gca, 'YTick', 1:topN);
xlabel('Importance Score');
title('Top 5 Most Important Audio Features');
grid on;
set(gca, 'YDir','reverse');  % Show most important at the top

% === Save the plot as image (optional) ===
saveas(gcf, 'top_features_importance.png');
