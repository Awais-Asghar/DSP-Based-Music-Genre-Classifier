% === Load image datastore ===
imds = imageDatastore('A:\DSP_project\mel_spectrograms\', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% === Define and enforce target genres ===
targetGenres = ["Folk", "Hip-Hop", "Instrumental", "International"];
imds.Labels = categorical(string(imds.Labels));
imds = subset(imds, ismember(imds.Labels, targetGenres));

% === Resize for CNN input ===
imds.ReadFcn = @(filename) imresize(imread(filename), [128 128]);

% === Display classes after filtering ===
disp("Filtered genres:");
disp(unique(imds.Labels));

% === Split into training/testing sets ===
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.9, 'randomized');

% === Data Augmentation ===
augmenter = imageDataAugmenter( ...
    'RandRotation', [-90 60], ...
    'RandXTranslation', [-45 55], ...
    'RandYTranslation', [-25 35], ...
    'RandXScale', [0.5 1.7], ...
    'RandYScale', [0.6 1.6]);

augimdsTrain = augmentedImageDatastore([128 128 1], imdsTrain, 'DataAugmentation', augmenter);
augimdsTest  = augmentedImageDatastore([128 128 1], imdsTest);

% === Count number of active classes ===
numClasses = numel(categories(imdsTrain.Labels));

% === CNN Architecture ===
layers = [
    imageInputLayer([128 128 1], 'Name', 'input')

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    dropoutLayer(0.2)
    globalAveragePooling2dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% === Training Options ===
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 2, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsTest, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'L2Regularization', 1e-4, ...
    'ExecutionEnvironment', 'auto', ...
    'LearnRateSchedule', 'piecewise');

% === Train Network ===
net = trainNetwork(augimdsTrain, layers, options);

% === Predict on Test Set ===
YPred_cnn = classify(net, augimdsTest);
YTrue_cnn = imdsTest.Labels;

% === Ensure consistent categories ===
YPred_cnn = categorical(string(YPred_cnn), targetGenres);
YTrue_cnn = categorical(string(YTrue_cnn), targetGenres);

% === Accuracy ===
acc = mean(YPred_cnn == YTrue_cnn);
fprintf('🎯 CNN Accuracy: %.2f%%\n', acc * 100);

% === Confusion Matrix ===
figure;
confusionchart(YTrue_cnn, YPred_cnn, ...
    'Title', 'CNN Confusion Matrix - 4 Genres', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');

% === Save model and predictions ===
save('genre_cnn_model_4genre.mat', 'net', 'YPred_cnn', 'YTrue_cnn');
