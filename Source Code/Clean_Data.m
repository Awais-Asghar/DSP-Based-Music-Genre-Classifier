% Load raw features and labels
load('fma_advanced_features_max5000.mat');  % featureList, genreLabels

% Step 1: Replace NaN and Inf values with feature-wise mean
featureMeans = mean(featureList, 'omitnan');
nanIdx = isnan(featureList) | isinf(featureList);

% Replace each NaN/Inf with corresponding column mean
for col = 1:size(featureList, 2)
    featureList(nanIdx(:, col), col) = featureMeans(col);
end

% Step 2: Handle missing or empty genre labels
for i = 1:length(genreLabels)
    if isempty(genreLabels{i}) || strcmp(genreLabels{i}, '') || strcmp(genreLabels{i}, 'NaN')
        genreLabels{i} = 'Unknown';
    end
end

% Optional: Convert to categorical for modeling
genreLabels = categorical(genreLabels);

% Save cleaned data
save('fma_cleaned_features.mat', 'featureList', 'genreLabels');
fprintf('✅ Cleaned data saved to fma_cleaned_features.mat\n');
