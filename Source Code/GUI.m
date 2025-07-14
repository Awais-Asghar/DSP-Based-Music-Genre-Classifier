function musicGenreClassifierGUI
    % Load models and genre metadata
    knnData = load('bestKNNModel_4Genres.mat');
    knnModel = knnData.bestKNNModel;
    svmData = load('trainedSVMModel_4Genres.mat');
    svmModel = svmData.SVMModel;
    metadata = readtable('A:/DSP_project/fma_metadata/tracks.csv');
    btData = load('trainedBoostedTreeModel.mat');
    btModel = btData.boostedTreeModel;
    cnnData = load('A:/DSP_project/Advanced/CNN/genre_cnn_model_4genre.mat');
    cnnModel = cnnData.net;

    player = [];  % Global player object for stopping audio
    allResults = {};  % To accumulate predictions

    % Opening splash screen
    splash = figure('Name', 'Welcome', 'NumberTitle', 'off', ...
        'Position', [500 300 400 250], 'Color', [0.1 0.1 0.2]);
    uicontrol(splash, 'Style', 'text', 'String', ...
        '🎵 Welcome to the Music Genre Classifier 🎵', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'ForegroundColor', 'white', 'BackgroundColor', [0.1 0.1 0.2], ...
        'Position', [20 170 360 50]);
    uicontrol(splash, 'Style', 'pushbutton', 'String', 'Continue', ...
        'FontSize', 12, 'Position', [150 30 100 30], ...
        'Callback', @(~,~) close(splash));
    waitfor(splash);

    % Create GUI
    f = figure('Name', 'Music Genre Classifier', ...
        'Position', [300 300 800 750], 'Color', [0.95 0.95 0.95]);

    % Filter metadata
    validSubset = strcmp(metadata.subset, 'small');
    filteredMeta = metadata(validSubset & ismember(metadata.genre_top, ...
        {'Folk', 'Hip-Hop', 'Instrumental', 'International'}), :);
    trackTitles = strcat(string(filteredMeta.track_id), " - ", filteredMeta.title);

    % GUI Controls
    uicontrol(f, 'Style', 'text', 'String', 'Or Choose from Dataset:', ...
        'FontSize', 11, 'Position', [30 595 180 25], ...
        'BackgroundColor', [0.95 0.95 0.95]);

    trackPopup = uicontrol(f, 'Style', 'popupmenu', 'String', trackTitles, ...
        'FontSize', 10, 'Position', [210 600 250 25]);

    uicontrol(f, 'Style', 'pushbutton', 'String', 'Classify Selected Track', ...
        'FontSize', 11, 'Position', [470 600 180 25], ...
        'Callback', @selectFromDropdown);

    uicontrol(f, 'Style', 'text', ...
        'String', 'Model Accuracies - KNN: 87.52% | SVM: 91.34% | BT: 89.73% | CNN: 93.12%', ...
        'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0.95 0.95 0.95], ...
        'Position', [30 660 690 20]);

    uicontrol(f, 'Style', 'pushbutton', 'String', 'Browse Files', ...
        'FontSize', 12, 'Position', [30 630 120 30], ...
        'Callback', @selectFiles);

    uicontrol(f, 'Style', 'pushbutton', 'String', 'Stop Music', ...
        'FontSize', 11, 'Position', [160 630 100 30], ...
        'Callback', @stopMusic);

    modelPopup = uicontrol(f, 'Style', 'popupmenu', ...
        'String', {'KNN', 'SVM', 'Both', 'Boosted Trees', 'CNN', 'All'}, ...
        'FontSize', 11, 'Position', [280 630 150 25]);

    uitable(f, 'Tag', 'resultTable', 'ColumnName', ...
        {'Track ID', 'Original Genre', 'KNN Genre', 'SVM Genre', 'BT Genre', 'CNN Genre'}, ...
        'ColumnWidth', {100 200 150 150 150 150}, ...
        'Position', [30 120 740 480], 'FontSize', 11);

    uicontrol(f, 'Style', 'pushbutton', 'String', 'Export to CSV', ...
        'FontSize', 11, 'Position', [640 630 120 30], ...
        'Callback', @exportResults);

    % Feature extractor
    afe = audioFeatureExtractor('SampleRate', 44100, ...
        'mfcc', true, 'SpectralCentroid', true, 'ZeroCrossRate', true, ...
        'SpectralRolloffPoint', true, 'SpectralFlux', true, 'SpectralEntropy', true, ...
        'SpectralSpread', true, 'SpectralSkewness', true, 'SpectralKurtosis', true, ...
        'SpectralCrest', true, 'harmonicRatio', true, 'pitch', true);
    setExtractorParameters(afe, 'mfcc', 'NumCoeffs', 13);
    setExtractorParameters(afe, 'pitch', 'Range', [50, 2000]);

    % === CALLBACKS ===
    function selectFiles(~, ~)
        modelChoice = get(modelPopup, 'Value');
        [files, path] = uigetfile({'*.mp3;*.wav', 'Audio Files'}, 'Select Audio Files', 'MultiSelect', 'on');
        if isequal(files, 0), return; end
        if ischar(files), files = {files}; end
        for i = 1:length(files)
            processAndDisplay(fullfile(path, files{i}), modelChoice);
        end
    end

    function selectFromDropdown(~, ~)
        idx = get(trackPopup, 'Value');
        trackID = filteredMeta.track_id(idx);
        folderName = sprintf('%03d', floor(trackID / 1000));
        fileName = sprintf('%06d.mp3', trackID);
        fullPath = fullfile('A:/DSP_project/fma_small/', folderName, fileName);
        modelChoice = get(modelPopup, 'Value');
        processAndDisplay(fullPath, modelChoice);
    end

function processAndDisplay(filePath, modelChoice)
    [~, name, ~] = fileparts(filePath);
    trackID = extractTrackID(name);
    genreRow = metadata(metadata.track_id == str2double(trackID) & ...
                        strcmp(metadata.subset, 'small'), :);
    originalGenre = 'Unknown';
    if ~isempty(genreRow)
        originalGenre = genreRow.genre_top{1};
    end

    try
        [audioIn, fs] = audioread(filePath);
        if size(audioIn, 2) > 1
            audioIn = mean(audioIn, 2);  % Convert to mono
        end
        targetLen = 30 * fs;
        audioIn = audioIn(1:min(end, targetLen));
        if length(audioIn) < targetLen
            audioIn = [audioIn; zeros(targetLen - length(audioIn), 1)];
        end

        feats = extract(afe, audioIn);
        meanFeats = mean(feats, 1, 'omitnan');

        cnnPred = 'N/A'; knnPred = ''; svmPred = ''; btPred = '';
        finalPred = '';

        if modelChoice == 5 || modelChoice == 6
            melSpec = melSpectrogram(audioIn, fs, ...
                'Windowlength', round(0.025 * fs), ...
                'OverlapLength', round(0.015 * fs), ...
                'NumBands', 128);

            melSpec = log10(melSpec + eps);
            melImg = rescale(melSpec);
            melImg = imresize(melImg, [128 128]);
            melImg = reshape(melImg, [128 128 1]);
            cnnPred = char(classify(cnnModel, melImg));
        end

        if modelChoice == 1 || modelChoice == 3 || modelChoice == 6
            knnPred = char(predict(knnModel, meanFeats));
        end
        if modelChoice == 2 || modelChoice == 3 || modelChoice == 6
            svmPred = char(predict(svmModel, meanFeats));
        end
        if modelChoice == 4 || modelChoice == 6
            btPred = char(predict(btModel, meanFeats));
        end

        % === Ensemble Prediction Logic ===
        if modelChoice == 3 || modelChoice == 6
            preds = {knnPred, svmPred, btPred, cnnPred};
            preds = preds(~strcmp(preds, ''));  % remove empty
            preds = preds(~strcmp(preds, 'N/A'));  % remove N/A
            % Majority voting
            uniqueGenres = unique(preds);
            counts = cellfun(@(g) sum(strcmp(preds, g)), uniqueGenres);
            [~, idx] = max(counts);
            finalPred = uniqueGenres{idx};
            
        else
            finalPred = 'N/A';
        end

        % === Play Audio ===
        player = audioplayer(audioIn, fs);
        play(player);

    catch ME
        warning('Error processing %s: %s', filePath, ME.message);
        knnPred = 'Error'; svmPred = 'Error'; btPred = 'Error'; cnnPred = 'Error';
        finalPred = 'Error';
    end

    % === Show Ensemble Prediction ===
    if modelChoice == 3 || modelChoice == 6
        msgbox(['🎧 Ensemble Prediction: ', finalPred], 'Final Genre', 'modal');
    end

    result = {char(trackID), char(originalGenre), char(knnPred), ...
              char(svmPred), char(btPred), char(cnnPred)};
    allResults = [allResults; result];
    resultTable = findobj(f, 'Tag', 'resultTable');
    set(resultTable, 'Data', allResults);
    setappdata(f, 'results', allResults);
end



    function stopMusic(~, ~)
        if ~isempty(player) && isplaying(player)
            stop(player);
        end
    end

    function exportResults(~, ~)
        results = getappdata(f, 'results');
        if isempty(results)
            errordlg('No results to export.');
            return;
        end
        [file, path] = uiputfile('genre_results.csv', 'Save Results');
        if file == 0, return; end
        writetable(cell2table(results, 'VariableNames', ...
            {'TrackID', 'OriginalGenre', 'KNNGenre', 'SVMGenre', 'BTGenre', 'CNNGenre'}), ...
            fullfile(path, file));
    end

    function id = extractTrackID(filename)
        digitsOnly = regexp(filename, '\d+', 'match');
        if ~isempty(digitsOnly)
            id = digitsOnly{1};
        else
            id = '0';
        end
    end
end
