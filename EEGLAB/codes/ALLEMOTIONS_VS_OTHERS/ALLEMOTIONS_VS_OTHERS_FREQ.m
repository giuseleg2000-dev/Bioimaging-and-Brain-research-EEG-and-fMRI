%% Emotion Comparison Analysis Script
% This script analyzes p-values from emotion comparison matrices and calculates frequency statistics
% of significant results across indices and channels

% Define emotion categories for comparison
emotionCategories = {'happy', 'sad', 'tender', 'angry', 'tense', 'pleasant', 'afraid', 'energetic'};

% Define file system paths
DATA_SOURCE_PATH = 'C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\MATRICI';
RESULTS_DESTINATION_PATH = 'C:\Users\giuseppe\OneDrive\Desktop\magistrale biomedical engineering engineering of medical devices\brain research\filetxt';

% Process all pairwise emotion comparisons
for primaryEmotionIdx = 1:length(emotionCategories)
    primaryEmotion = emotionCategories{primaryEmotionIdx};
    
    for secondaryEmotionIdx = 1:length(emotionCategories)
        secondaryEmotion = emotionCategories{secondaryEmotionIdx};
        
        % Skip self-comparisons
        if primaryEmotionIdx == secondaryEmotionIdx
            continue;
        end
        
        % Construct the input file path
        inputFileName = sprintf('Pvalues_%s_vs_%s.mat', primaryEmotion, secondaryEmotion);
        fullInputPath = fullfile(DATA_SOURCE_PATH, inputFileName);
        
        % Process the file if it exists
        if exist(fullInputPath, 'file')
            % Load matrix data
            matData = load(fullInputPath);
            
            % Extract the p-value matrix (first field in the loaded structure)
            fieldList = fieldnames(matData);
            pValueMatrix = matData.(fieldList{1});
            
            % Find positions with significant p-values (value = 1)
            [channelIndices, frequencyIndices] = find(pValueMatrix == 1);
            
            % Calculate statistics
            totalSignificantPoints = length(frequencyIndices);
            
            % Analyze frequency distributions across indices
            [indexCounts, uniqueIndices] = groupcounts(frequencyIndices);
            relativeFreqIndices = round((indexCounts / totalSignificantPoints) * 100, 2);
            
            % Analyze frequency distributions across channels
            [channelCounts, uniqueChannels] = groupcounts(channelIndices);
            relativeFreqChannels = round((channelCounts / totalSignificantPoints) * 100, 2);
            
            % Save index frequency results
            indexOutputFileName = sprintf('frequenza_Indice_%s_vs_%s.txt', primaryEmotion, secondaryEmotion);
            indexOutputPath = fullfile(RESULTS_DESTINATION_PATH, indexOutputFileName);
            indexResultsTable = table(uniqueIndices, relativeFreqIndices, 'VariableNames', {'Indice', 'Frequenza_Indice'});
            writetable(indexResultsTable, indexOutputPath, 'Delimiter', '\t', 'WriteVariableNames', true);
            
            % Save channel frequency results
            channelOutputFileName = sprintf('frequenza_Canale_%s_vs_%s.txt', primaryEmotion, secondaryEmotion);
            channelOutputPath = fullfile(RESULTS_DESTINATION_PATH, channelOutputFileName);
            channelResultsTable = table(uniqueChannels, relativeFreqChannels, 'VariableNames', {'Canale', 'Frequenza_Canale'});
            writetable(channelResultsTable, channelOutputPath, 'Delimiter', '\t', 'WriteVariableNames', true);
            
            % Log successful operations
            fprintf('Analysis complete: %s vs %s\n', primaryEmotion, secondaryEmotion);
            fprintf('  - Index results saved to: %s\n', indexOutputFileName);
            fprintf('  - Channel results saved to: %s\n', channelOutputFileName);
        else
            fprintf('Warning: Input file not found - %s\n', inputFileName);
        end
    end
end

fprintf('Analysis completed for all emotion comparisons.\n');