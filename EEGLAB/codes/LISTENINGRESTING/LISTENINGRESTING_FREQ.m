%% LISTENING_RESTING Analysis Script
% This script analyzes a matrix from LISTENING_RESTING.mat file
% and calculates frequency distributions of matrix elements with value 1

% Define emotion labels (unused in current script but kept for reference)
emotions = {'happy', 'sad', 'tender', 'angry', 'tense', 'pleasant', 'afraid', 'energetic'};

%% Load and Process Data
% Define file paths
input_filepath = 'C:\Users\giuseppe\OneDrive\Desktop\magistrale biomedical engineering engineering of medical devices\brain research\MATRICI\LISTENING_RESTING.mat';
output_dir = 'C:\Users\giuseppe\OneDrive\Desktop\magistrale biomedical engineering engineering of medical devices\brain research\filetxt\';
index_output = fullfile(output_dir, 'LRfrequenza_Indice.txt');
channel_output = fullfile(output_dir, 'LRfrequenza_canali.txt');

% Load the matrix data
data = load(input_filepath);
varNames = fieldnames(data);
matrix = data.(varNames{1});  % Use the first variable in the file

%% Find Positions with Value 1
[row, col] = find(matrix == 1);
total_ones = length(col);

% Display results
disp('Positions of 1s in the matrix:');
results_table = table(col, row);
disp(results_table);

%% Calculate Column Frequencies
[counts_cols, col_values] = groupcounts(col);
freq_cols = (counts_cols / total_ones) * 100;
freq_cols = round(freq_cols, 2);  % Round to 2 decimal places

% Display column statistics
col_table = table(col_values, counts_cols, freq_cols, ...
                 'VariableNames', {'Index', 'Count', 'Relative_Frequency'});
disp('Count and relative frequency of each column:');
disp(col_table);

%% Calculate Row Frequencies
[counts_rows, row_values] = groupcounts(row);
freq_rows = (counts_rows / total_ones) * 100;
freq_rows = round(freq_rows, 2);  % Round to 2 decimal places

% Display row statistics
row_table = table(row_values, counts_rows, freq_rows, ...
                 'VariableNames', {'Channel', 'Count', 'Relative_Frequency'});
disp('Count and relative frequency of each row:');
disp(row_table);

%% Save Results to Files
% Save index frequencies
index_output_table = table(col_values, freq_cols, 'VariableNames', {'Index', 'Frequency_Index'});
writetable(index_output_table, index_output, 'Delimiter', '\t', 'WriteVariableNames', true);
disp(['The index data have been saved in the file: ', index_output]);

% Save channel frequencies
channel_output_table = table(row_values, freq_rows, 'VariableNames', {'Channel', 'Frequency_Channel'});
writetable(channel_output_table, channel_output, 'Delimiter', '\t', 'WriteVariableNames', true);
disp(['The channel data have been saved in the file: ', channel_output]);