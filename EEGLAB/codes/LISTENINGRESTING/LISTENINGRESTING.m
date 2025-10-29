% EMOTION RECOGNITION FROM EEG SIGNALS
% This code processes EEG data to recognize 8 emotions and 8 intensities
% and assigns the most intense emotion to each segment

% Loop through all subjects
for subject_id = 12:21
    
    % Process both runs for each subject
    for run_id = 1:2
        
        % === DATA LOADING AND INITIALIZATION ===
        % Load EEG data from .edf file
        file_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\SUBJECTS\\sub-%d\\eeg\\sub-%d_task-run%d_eeg.edf', subject_id, subject_id, run_id);
        EEG = pop_biosig(file_path);
        
        % Load events from .tsv file
        events_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\SUBJECTS\\sub-%d\\eeg\\sub-%d_task-run%d_events.tsv', subject_id, subject_id, run_id);
        EEG.event = readtable(events_path, 'FileType', 'text', 'Delimiter', '\t');
        
        % Load channel locations
        EEG.chanlocs = struct('labels', {'FP1' 'FP2' 'F7' 'F3' 'Fz' 'F4' 'F8' 'T7' 'C3' 'Cz' 'C4' 'T8' 'P7' 'P3' 'Pz' 'P4' 'P8' 'O1' 'O2'}, 'ref', [],'theta',[], 'radius',[], 'X',[], 'Y',[], 'Z',[], 'sph_theta',[], 'sph_phi',[], 'sph_radius',[], 'type', [],'urchan',[]);
        EEG.chanlocs = readlocs('D:\desktop personal\EEGLAB\eeglab_current\eeglab2023.1\functions\supportfiles\channel_location_files\eeglab\Standard-10-20-Cap19.ced');
        
        % === SIGNAL PREPROCESSING ===
        % Resample data to 500 Hz
        target_fs = 500;
        EEG = pop_resample(EEG, target_fs);
        
        % Create a preprocessed copy of the data
        preprocessed_EEG = EEG;
        
        % Apply bandpass filter (0.5-80 Hz)
        [b_bandpass, a_bandpass] = butter(3, [0.5, 80]/(target_fs/2), 'bandpass');
        preprocessed_EEG.data = filter(b_bandpass, a_bandpass, preprocessed_EEG.data);
        
        % Apply CleanLine for 50Hz noise removal (three passes for better results)
        for cleanline_pass = 1:3
            preprocessed_EEG = pop_cleanline(preprocessed_EEG, ...
                'Bandwidth', 1, ...
                'ChanCompIndices', 1:preprocessed_EEG.nbchan, ...
                'SignalType', 'Channels', ...
                'ComputeSpectralPower', true, ...
                'LineFrequencies', 50, ... % Line noise at 50Hz
                'NormalizeSpectrum', false, ...
                'LineAlpha', 0.01, ...
                'PaddingFactor', 2, ...
                'ScanForLines', false, ...
                'PlotFigures', false, ...
                'SlidingWinLength', 4, ...
                'SlidingWinStep', 2, ...
                'VerbosityLevel', 1);
        end
        
        % Apply average reference
        channel_avg = mean(preprocessed_EEG.data, 1);
        preprocessed_EEG.data = preprocessed_EEG.data - channel_avg;
        
        % === INDEPENDENT COMPONENT ANALYSIS (ICA) ===
        % Run ICA with extended infomax algorithm
        preprocessed_EEG = pop_runica(preprocessed_EEG, 'extended', 1, 'maxsteps', 30);
        
        % Apply ICLabel to classify components
        preprocessed_EEG = iclabel(preprocessed_EEG);
        
        % Extract component classifications
        component_classes = preprocessed_EEG.etc.ic_classification.ICLabel.classifications;
        class_labels = preprocessed_EEG.etc.ic_classification.ICLabel.classes;
        
        % Set thresholds for component classification
        brain_threshold = 0.5;
        other_threshold = 0.5;
        
        % Find brain and non-brain components
        brain_class_idx = find(strcmp(class_labels, 'Brain'));
        other_class_indices = setdiff(1:length(class_labels), brain_class_idx);
        
        % Initialize component lists
        brain_components = [];
        non_brain_components = [];
        
        % Classify components as brain or non-brain
        for comp_idx = 1:size(component_classes, 1)
            brain_prob = component_classes(comp_idx, brain_class_idx);
            other_probs = component_classes(comp_idx, other_class_indices);
            
            if brain_prob >= brain_threshold
                % Brain probability >= threshold
                brain_components = [brain_components, comp_idx];
            elseif brain_prob < brain_threshold && all(other_probs < other_threshold)
                % Brain probability < threshold but all others also below threshold
                brain_components = [brain_components, comp_idx];
            else
                % Non-brain component
                non_brain_components = [non_brain_components, comp_idx];
            end
        end
        
        % Remove non-brain components
        preprocessed_EEG = pop_subcomp(preprocessed_EEG, non_brain_components, 0);
        
        % Display results of component removal
        disp('Identified "Brain" components:');
        disp(brain_components);
        disp('Removed "Non-Brain" components:');
        disp(non_brain_components);
        
        % Recalculate ICA activity with remaining components
        preprocessed_EEG.icaact = (preprocessed_EEG.icaweights * preprocessed_EEG.icasphere) * preprocessed_EEG.data(preprocessed_EEG.icachansind, :);
        
        % Reconstruct EEG data using only brain components
        preprocessed_EEG.data = preprocessed_EEG.icawinv * preprocessed_EEG.icaact;
        
        % === SEGMENT PROCESSING ===
        % Convert events to structure if needed
        if istable(preprocessed_EEG.event)
            preprocessed_EEG.event = table2struct(preprocessed_EEG.event);
        elseif ~isstruct(preprocessed_EEG.event)
            error('preprocessed_EEG.event is neither a table nor a structure. Check the input data.');
        end
        
        % Process musical segments in run 2
        if run_id == 2
            % Adapt event fields for EEGLAB
            for evt_idx = 1:length(preprocessed_EEG.event)
                preprocessed_EEG.event(evt_idx).latency = preprocessed_EEG.event(evt_idx).onset * preprocessed_EEG.srate;
                preprocessed_EEG.event(evt_idx).type = preprocessed_EEG.event(evt_idx).trial_type;
            end
            
            % Define emotion labels and codes
            emotion_labels = {'pleasant', 'energetic', 'tense', 'angry', 'afraid', 'happy', 'sad', 'tender'};
            emotion_codes = 800:807;
            intensity_codes = 901:909;
            
            % Find segment start markers (code 786)
            segment_indices = find([preprocessed_EEG.event.type] == 786);
            num_segments = length(segment_indices);
            
            % Process each segment
            for seg_idx = 1:num_segments
                % Find segment boundaries
                start_idx = segment_indices(seg_idx);
                
                if seg_idx < num_segments
                    end_idx = segment_indices(seg_idx + 1) - 1;
                else
                    end_idx = length(preprocessed_EEG.event);
                end
                
                % Initialize emotion tracking
                emotion_list = {};
                intensity_list = [];
                
                % Analyze events in this segment
                for evt_idx = start_idx+1:end_idx
                    event_code = preprocessed_EEG.event(evt_idx).type;
                    
                    % Check if event is an emotion code
                    if ismember(event_code, emotion_codes)
                        emotion_idx = event_code - 799;
                        emotion = emotion_labels{emotion_idx};
                        
                        % Look for corresponding intensity
                        intensity = NaN;
                        for intensity_idx = evt_idx+1:end_idx
                            intensity_code = preprocessed_EEG.event(intensity_idx).type;
                            
                            if ismember(intensity_code, intensity_codes)
                                intensity = intensity_code - 900;
                                break;
                            end
                        end
                        
                        % Store emotion and intensity if not already present
                        if ~ismember(emotion, emotion_list)
                            emotion_list{end+1} = emotion;
                            intensity_list(end+1) = intensity;
                        end
                    end
                end
                
                % Determine dominant emotion based on intensity
                [max_intensity, max_idx] = max(intensity_list);
                if ~isempty(max_idx)
                    dominant_emotion = emotion_list{max_idx};
                else
                    dominant_emotion = 'undefined';
                end
                
                % Extract EEG data for music period (code 788)
                music_start_idx = find([preprocessed_EEG.event(start_idx:end_idx).type] == 788, 1, 'first');
                if ~isempty(music_start_idx)
                    music_start_sample = preprocessed_EEG.event(start_idx + music_start_idx - 1).latency;
                    
                    % Extract 12 seconds (6000 samples at 500 Hz)
                    music_end_sample = music_start_sample + 6000 - 1;
                    music_end_sample = min(music_end_sample, size(preprocessed_EEG.data, 2));
                    
                    segment_data = pop_select(preprocessed_EEG, 'point', [music_start_sample, music_end_sample]);
                else
                    segment_data = [];
                end
                
                % Save segment results
                save_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment%d.mat', subject_id, run_id, seg_idx);
                save(save_path, 'emotion_list', 'intensity_list', 'dominant_emotion', 'segment_data');
                fprintf('Complete results for segment %d saved in %s\n', seg_idx, save_path);
                
                % Display dominant emotion
                fprintf('Segment %d: Dominant emotion = %s (Intensity = %d)\n', seg_idx, dominant_emotion, max_intensity);
            end
        end
        
        % Process resting state in run 1
        if run_id == 1
            % Convert events if needed
            if istable(preprocessed_EEG.event)
                preprocessed_EEG.event = table2struct(preprocessed_EEG.event);
            end
            
            % Add latency field if missing
            if ~isfield(preprocessed_EEG.event, 'latency')
                for evt_idx = 1:length(preprocessed_EEG.event)
                    if isfield(preprocessed_EEG.event, 'onset')
                        preprocessed_EEG.event(evt_idx).latency = preprocessed_EEG.event(evt_idx).onset * preprocessed_EEG.srate;
                    else
                        error('preprocessed_EEG.event is missing the "latency" or "onset" field.');
                    end
                end
            end
            
            % Extract segment from the end of run 1
            total_samples = size(preprocessed_EEG.data, 2);
            start_sample = total_samples - 10000;
            segment_length = round(12 * preprocessed_EEG.srate); % 12 seconds
            end_sample = min(start_sample + segment_length - 1, total_samples);
            
            % Validate sample range
            if start_sample < 1 || end_sample > total_samples
                error('Invalid sample range.');
            end
            
            % Extract the segment
            segment_data = pop_select(preprocessed_EEG, 'point', [start_sample, end_sample]);
            
            % Save the segment
            save_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment1.mat', subject_id, run_id);
            save(save_path, 'segment_data');
            fprintf('Saved single segment in %s\n', save_path);
        end
        
        % === SPECTRAL POWER ANALYSIS ===
        % Define frequency bands
        freq_bands = struct(...
            'delta', [0.5 4], ...
            'theta', [4 8], ...
            'alpha', [8 12], ...
            'beta', [12 35], ...
            'gamma', [35 80], ...
            'smr', [12 15]);
        
        % Number of channels and FFT points
        num_channels = size(preprocessed_EEG.data, 1);
        nfft = 2^nextpow2(size(preprocessed_EEG.data, 2));
        
        % Initialize power arrays
        band_power = struct(...
            'delta', zeros(num_channels, 1), ...
            'theta', zeros(num_channels, 1), ...
            'alpha', zeros(num_channels, 1), ...
            'beta', zeros(num_channels, 1), ...
            'gamma', zeros(num_channels, 1), ...
            'smr', zeros(num_channels, 1));
        
        % Initialize indices matrix
        I = zeros(num_channels, 37);
        
        % Process segments based on run
        if run_id == 2
            % Process all segments in run 2
            for seg_idx = 1:num_segments
                % Load segment data
                segment_file = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment%d.mat', subject_id, run_id, seg_idx);
                segment_data = load(segment_file, 'segment_data', 'dominant_emotion');
                
                % Initialize index matrix
                num_segment_channels = size(segment_data.segment_data.data, 1);
                I = zeros(num_segment_channels, 37);
                nfft = 2^nextpow2(size(segment_data.segment_data.data, 2));
                
                % Calculate power spectrum for each channel
                for ch_idx = 1:num_segment_channels
                    signal = segment_data.segment_data.data(ch_idx, :);
                    [psd, freqs] = pwelch(signal, [], [], nfft, target_fs);
                    
                    % Calculate power in each frequency band
                    for band = fieldnames(freq_bands)'
                        band_name = band{1};
                        band_range = freq_bands.(band_name);
                        band_idx = freqs >= band_range(1) & freqs <= band_range(2);
                        band_power.(band_name)(ch_idx) = sum(psd(band_idx));
                    end
                    
                    % Calculate all 37 indices for this channel
                    I(ch_idx, :) = calculateIndices(band_power, ch_idx);
                end
                
                % Save results
                results_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment%d_results.mat', subject_id, run_id, seg_idx);
                save(results_path, 'I', 'segment_data', 'dominant_emotion');
                fprintf('Complete results for segment %d saved in %s\n', seg_idx, results_path);
            end
        else
            % Process single segment in run 1
            segment_file = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment1.mat', subject_id, run_id);
            segment_data = load(segment_file, 'segment_data');
            
            % Calculate power spectrum for each channel
            num_segment_channels = size(segment_data.segment_data.data, 1);
            I = zeros(num_segment_channels, 37);
            nfft = 2^nextpow2(size(segment_data.segment_data.data, 2));
            
            for ch_idx = 1:num_segment_channels
                signal = segment_data.segment_data.data(ch_idx, :);
                [psd, freqs] = pwelch(signal, [], [], nfft, target_fs);
                
                % Calculate power in each frequency band
                for band = fieldnames(freq_bands)'
                    band_name = band{1};
                    band_range = freq_bands.(band_name);
                    band_idx = freqs >= band_range(1) & freqs <= band_range(2);
                    band_power.(band_name)(ch_idx) = sum(psd(band_idx));
                end
                
                % Calculate all 37 indices for this channel
                I(ch_idx, :) = calculateIndices(band_power, ch_idx);
            end
            
            % Save results
            results_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment1_results.mat', subject_id, run_id);
            save(results_path, 'I', 'segment_data');
            fprintf('Complete results saved in %s\n', results_path);
        end
    end
end

% === STATISTICAL ANALYSIS ===
% Initialize statistical results matrices
p_values = zeros(19, 37);
h_values = zeros(19, 37);

% Loop through all channels and indices
for channel = 1:19
    for index = 1:37
        % Initialize data arrays
        music_data = [];
        resting_data = [];
        
        % Collect data across all subjects and segments
        for subject = 12:21
            % Process 10 segments for music condition
            for segment = 1:10
                % Load music condition data (run 2)
                music_file = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run2_segment%d_results.mat', subject, segment);
                music_result = load(music_file, 'I');
                
                % Load resting condition data (run 1)
                resting_file = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run1_segment1_results.mat', subject);
                resting_result = load(resting_file, 'I');
                
                % Add data for statistical analysis
                music_data = [music_data, music_result.I(channel, index)];
                resting_data = [resting_data, resting_result.I(channel, index)];
            end
        end
        
        % Perform Wilcoxon signed-rank test
        [p, h] = signrank(music_data, resting_data);
        
        % Store results
        p_values(channel, index) = p;
        h_values(channel, index) = h;
    end
end

% Save statistical results
save('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\statistical_results_subjects.mat', 'p_values', 'h_values');
save('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\MATRICI\LISTENING_RESTING.mat', 'p_values', 'h_values');

% Visualize p-value matrix
figure;
imagesc(p_values);
colorbar;
caxis([0.01 0.1]);
title('P-values Wilcoxon Test (Listening vs Resting)');
xlabel('EEG Indices');
ylabel('EEG Channels');

% === HELPER FUNCTIONS ===
function indices = calculateIndices(band_power, ch_idx)
    % Helper function to calculate all 37 indices
    % This keeps the main code cleaner by encapsulating the calculations
    
    % Extract powers for this channel
    alpha = band_power.alpha(ch_idx);
    beta = band_power.beta(ch_idx);
    delta = band_power.delta(ch_idx);
    theta = band_power.theta(ch_idx);
    gamma = band_power.gamma(ch_idx);
    smr = band_power.smr(ch_idx);
    
    % Calculate all 37 indices
    indices = zeros(1, 37);
    
    indices(1) = beta / alpha; % I1
    indices(2) = beta / (alpha + theta); % I2
    indices(3) = beta / theta; % I3
    indices(4) = theta / alpha; % I4
    indices(5) = theta / delta; % I5
    indices(6) = smr / theta; % I6
    indices(7) = smr / beta; % I7
    indices(8) = (alpha + beta) / delta; % I8
    indices(9) = (theta + alpha) / (alpha + beta); % I9
    indices(10) = theta / (alpha + beta); % I10
    indices(11) = (theta + alpha) / gamma; % I11
    indices(12) = (beta + theta) / alpha; % I12
    indices(13) = (delta + theta) / beta; % I13
    indices(14) = (delta + theta + alpha) / beta; % I14
    indices(15) = (delta + theta) / alpha; % I15
    indices(16) = (delta + theta) / (alpha + beta); % I16
    indices(17) = delta / alpha; % I17
    indices(18) = delta / beta; % I18
    indices(19) = theta / gamma; % I19
    indices(20) = alpha / gamma; % I20
    indices(21) = (smr + beta) / theta; % I21
    indices(22) = (theta + alpha) / (beta + gamma); % I22
    indices(23) = (alpha + beta) / (alpha + theta); % I23
    indices(24) = alpha / (beta + gamma); % I24
    indices(25) = (delta + theta + alpha) / (beta + gamma); % I25
    indices(26) = alpha / (delta + theta + alpha); % I26
    indices(27) = alpha / (theta + alpha + beta); % I27
    indices(28) = beta / (theta + gamma); % I28
    indices(29) = (beta + gamma) / delta; % I29
    indices(30) = (alpha + beta) / gamma; % I30
    indices(31) = (alpha + gamma) / (theta + delta); % I31
    indices(32) = (theta + alpha) / delta; % I32
    indices(33) = (theta + beta) / (alpha + gamma); % I33
    indices(34) = (beta + gamma) / (delta + theta); % I34
    indices(35) = (delta + alpha) / (theta + gamma); % I35
    indices(36) = (theta + alpha) / (delta + beta + gamma); % I36
    indices(37) = (alpha + beta) / (delta + theta + gamma); % I37
end