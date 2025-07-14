% Advanced Feature Extraction for fma_small Dataset with 5000 max per genre
audioDir = 'A:\DSP_project\fma_small\';
metadataFile = 'A:\DSP_project\fma_metadata\tracks.csv';

% Load metadata
opts = detectImportOptions(metadataFile);
metadata = readtable(metadataFile, opts);

% Filter only 'small' subset
smallSubset = strcmp(metadata.subset, 'small');
trackIDs = metadata.track_id(smallSubset);
genres = metadata.genre_top(smallSubset);

% Initialize genre counters
genreCounts = containers.Map();  % Tracks how many files per genre

% Output variables
featureList = [];
genreLabels = [];

% Audio Feature Extractor
afe = audioFeatureExtractor( ...
    'SampleRate', 44100, ...
    'mfcc', true, ...
    'SpectralCentroid', true, ...
    'ZeroCrossRate', true, ...
    'SpectralRolloffPoint', true, ...
    'SpectralFlux', true, ...
    'SpectralEntropy', true, ...
    'SpectralSpread', true, ...
    'SpectralSkewness', true, ...
    'SpectralKurtosis', true, ...
    'SpectralCrest', true, ...
    'harmonicRatio', true, ...
    'pitch', true);

setExtractorParameters(afe, 'mfcc', 'NumCoeffs', 13);
%setExtractorParameters(afe, 'SpectralRolloffPoint', 'RolloffPercentage', 0.85);
setExtractorParameters(afe, 'pitch', 'Range', [50, 2000]);

% Maximum clips per genre
maxPerGenre = 500;

% Loop through all tracks
for i = 1:length(trackIDs)
    try
        trackID = trackIDs(i);
        genre = string(genres{i});  % Safe conversion

        % Genre skip if limit reached
        if isKey(genreCounts, genre)
            if genreCounts(genre) >= maxPerGenre
                continue;
            end
        else
            genreCounts(genre) = 0;
        end

        % Construct file path
        folderName = sprintf('%03d', floor(trackID / 1000));
        fileName = sprintf('%06d.mp3', trackID);
        filePath = fullfile(audioDir, folderName, fileName);

        if ~isfile(filePath)
            fprintf('❌ File missing: %s\n', filePath);
            continue;
        end

        % Read audio
        [audioIn, fs] = audioread(filePath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);
        end

        % Standardize length to 30s
        targetLen = 30 * fs;
        audioIn = audioIn(1:min(end, targetLen));
        if length(audioIn) < targetLen
            audioIn = [audioIn; zeros(targetLen - length(audioIn), 1)];
        end

        % Extract features
        feats = extract(afe, audioIn);
        meanFeats = mean(feats, 1, 'omitnan');

        % Append data
        featureList = [featureList; meanFeats];
        genreLabels = [genreLabels; genre];
        genreCounts(genre) = genreCounts(genre) + 1;

        fprintf('✅ %d [%s] (%d/%d)\n', trackID, genre, genreCounts(genre), maxPerGenre);

    catch ME
        fprintf('⚠️ Error with track %d: %s\n', trackID, ME.message);
        continue;
    end
end

% Optional: Normalize features for better KNN performance
% featureList = normalize(featureList);

% Save extracted data
save('fma_advanced_features_max5000.mat', 'featureList', 'genreLabels');
fprintf('\n🎉 Saved to fma_advanced_features_max5000.mat\n');
