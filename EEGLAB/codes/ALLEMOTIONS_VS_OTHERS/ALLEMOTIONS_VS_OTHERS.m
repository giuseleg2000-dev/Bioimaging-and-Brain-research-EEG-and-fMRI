for z=12:21 % subjects in the group
    for j=1:2 % runs of the subjects
        % Read the raw EEG .edf file and related events
        % Base path for file access
        base_path = 'C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\';
        
        % Read the raw EEG .edf file
        EEG = pop_biosig(sprintf('%sSUBJECTS\\sub-%d\\eeg\\sub-%d_task-run%d_eeg.edf', base_path, z, z, j));
        
        % Read the events .tsv file
        EEG.event = readtable(sprintf('%sSUBJECTS\\sub-%d\\eeg\\sub-%d_task-run%d_events.tsv', base_path, z, z, j), 'FileType', 'text', 'Delimiter', '\t');
        
        %EEG.chanlocs = readlocs('C:\Users\LENOVO\Desktop\eeglab2024.2\functions\supportfiles\Standard-10-5-Cap385.sfp');  % Standard channel locations file
        
        % Define channel locations
        EEG.chanlocs = struct('labels', {'FP1' 'FP2' 'F7' 'F3' 'Fz' 'F4' 'F8' 'T7' 'C3' 'Cz' 'C4' 'T8' 'P7' 'P3' 'Pz' 'P4' 'P8' 'O1' 'O2'}, ...
                             'ref', [], 'theta', [], 'radius', [], 'X', [], 'Y', [], 'Z', [], ...
                             'sph_theta', [], 'sph_phi', [], 'sph_radius', [], 'type', [], 'urchan', []);
        
        EEG.chanlocs = readlocs('D:\desktop personal\EEGLAB\eeglab_current\eeglab2023.1\functions\supportfiles\channel_location_files\eeglab\Standard-10-20-Cap19.ced');
        
        % pop_chanedit(EEG.chanlocs);
        
        % PREPROCESSING SECTION
        % ---------------------
        % Define the new sampling frequency
        fs = 500;
        
        % Resample data to the target frequency
        EEG = pop_resample(EEG, fs);
        
        % BUTTERWORTH FILTER PARAMETERS
        order = 4;            % Filter order
        f_low = 0.5;          % Lower cutoff frequency in Hz
        f_high = 80;          % Upper cutoff frequency in Hz - clinical EEG range 0.5-80Hz
        f0 = 50;              % Powerline frequency to eliminate
        bw = 1;               % Filter bandwidth
        
        % Design bandpass filter
        [g, h] = butter(3, [0.5, 80]/(fs/2), 'bandpass');
        
        % Apply the filter to the data
        SEGN = EEG;
        SEGN.data = filter(g, h, SEGN.data);
        
        % Apply CLEANLINE three times to ensure complete removal of line noise
        % First application
        SEGN = pop_cleanline(SEGN, ...
                         'Bandwidth', 1, ...
                         'ChanCompIndices', 1:SEGN.nbchan, ...
                         'SignalType', 'Channels', ...
                         'ComputeSpectralPower', true, ...
                         'LineFrequencies', 50, ... % Powerline noise at 50Hz
                         'NormalizeSpectrum', false, ...
                         'LineAlpha', 0.01, ...
                         'PaddingFactor', 2, ...
                         'ScanForLines', false, ...
                         'PlotFigures', false, ... 
                         'SlidingWinLength', 4, ...
                         'SlidingWinStep', 2, ...
                         'VerbosityLevel', 1);
        
        % Second application
        SEGN = pop_cleanline(SEGN, ...
                         'Bandwidth', 1, ...
                         'ChanCompIndices', 1:SEGN.nbchan, ...
                         'SignalType', 'Channels', ...
                         'ComputeSpectralPower', true, ...
                         'LineFrequencies', 50, ... 
                         'NormalizeSpectrum', false, ...
                         'LineAlpha', 0.01, ...
                         'PaddingFactor', 2, ...
                         'ScanForLines', false, ...
                         'PlotFigures', false, ... 
                         'SlidingWinLength', 4, ...
                         'SlidingWinStep', 2, ...
                         'VerbosityLevel', 1);
        
        % Third application
        SEGN = pop_cleanline(SEGN, ...
                         'Bandwidth', 1, ...
                         'ChanCompIndices', 1:SEGN.nbchan, ...
                         'SignalType', 'Channels', ...
                         'ComputeSpectralPower', true, ...
                         'LineFrequencies', 50, ... 
                         'NormalizeSpectrum', false, ...
                         'LineAlpha', 0.01, ...
                         'PaddingFactor', 2, ...
                         'ScanForLines', false, ...
                         'PlotFigures', false, ... 
                         'SlidingWinLength', 4, ...
                         'SlidingWinStep', 2, ...
                         'VerbosityLevel', 1);
        
        % Apply average reference to normalize the signals
        average_ref = mean(SEGN.data, 1);
        SEGN.data = SEGN.data - average_ref;
        
        % Run ICA decomposition with extended infomax algorithm
        SEGN = pop_runica(SEGN, 'extended', 1, 'maxsteps', 30);
        
        % Get the number of ICA components
        num_components = size(SEGN.icaweights, 1);
        
        % Visualize ICA components topography
        pop_topoplot(SEGN, 0, 1:num_components, 'ICA Components', 0, 'electrodes', 'on');
        
        % Apply ICLabel to classify ICA components
        SEGN = iclabel(SEGN);
        
        % Extract component classifications and class labels
        classifications = SEGN.etc.ic_classification.ICLabel.classifications;
        classes = SEGN.etc.ic_classification.ICLabel.classes;
        
        % Verify that ICA weights have been calculated
        disp(SEGN.icaweights);
        
        % Visualize the ICA component maps for selection
        pop_selectcomps(SEGN, [1:size(SEGN.icaweights, 1)]);
        
        % Define thresholds for component classification
        threshold_brain = 0.5;  % Threshold for brain components
        threshold_others = 0.5; % Threshold for non-brain components
        
        % Find the index of the "Brain" class
        brain_class_idx = find(strcmp(classes, 'Brain'));
        
        % Find indices of other (non-brain) classes
        other_class_indices = setdiff(1:length(classes), brain_class_idx);
        
        % Initialize lists for brain and non-brain components
        brain_components = [];
        non_brain_components = [];
        
        % Loop through all ICA components to classify them
        for i = 1:size(classifications, 1)
            brain_prob = classifications(i, brain_class_idx);
            other_probs = classifications(i, other_class_indices);
            
            % Apply classification rules
            if brain_prob >= threshold_brain
                % If "Brain" probability >= threshold
                brain_components = [brain_components, i];
            elseif brain_prob < threshold_brain && all(other_probs < threshold_others)
                % If "Brain" < threshold but all other components < threshold
                brain_components = [brain_components, i];
            elseif brain_prob < threshold_brain && any(other_probs >= threshold_others)
                % If "Brain" < threshold and any other component >= threshold
                non_brain_components = [non_brain_components, i];
            end
        end
        
        % Remove non-brain components from the dataset
        SEGN = pop_subcomp(SEGN, non_brain_components, 0);
        
        % Display classification results
        disp('Identified "Brain" components:');
        disp(brain_components);
        disp('Removed "Non-Brain" components:');
        disp(non_brain_components);
        
        % Calculate ICA activations with remaining brain components
        SEGN.icaact = (SEGN.icaweights * SEGN.icasphere) * SEGN.data(SEGN.icachansind, :);
        
        % Reconstruct EEG signals in original channels using remaining components
        SEGN.data = SEGN.icawinv * SEGN.icaact;
        
        % END OF PREPROCESSING
        
        % EVENT PROCESSING SECTION
        % -----------------------
        % Check the type of SEGN.event and convert to structure if needed
        if istable(SEGN.event)
            SEGN.event = table2struct(SEGN.event);
        elseif ~isstruct(SEGN.event)
            error('SEGN.event is neither a table nor a structure. Check input data.');
        end
        
        % Process events for run 2 - contains emotion data
        if j == 2
            % Ensure structure fields are compatible with EEGLAB
            for i = 1:length(SEGN.event)
                SEGN.event(i).latency = SEGN.event(i).onset * SEGN.srate;
                SEGN.event(i).type = SEGN.event(i).trial_type;
            end
            
            % Define emotion labels and codes
            emotion_labels = {'pleasant', 'energetic', 'tense', 'angry', ...
                             'afraid', 'happy', 'sad', 'tender'};
            emotion_codes = 800:807;
            
            % Define intensity codes
            intensity_codes = 901:909;
            
            % Find segment markers (code 786)
            segment_indices = find([SEGN.event.type] == 786);
            num_segments = length(segment_indices);
            
            % Process each segment
            for seg_idx = 1:num_segments
                % Determine segment boundaries
                start_idx = segment_indices(seg_idx);
                
                if seg_idx < num_segments
                    end_idx = segment_indices(seg_idx + 1) - 1;
                else
                    end_idx = length(SEGN.event);
                end
                
                % Initialize emotion and intensity lists
                emotion_list = {};
                intensity_list = [];
                
                % Scan events in the current segment
                for i = start_idx+1:end_idx
                    code = SEGN.event(i).type;
                    
                    % Process emotion codes
                    if ismember(code, emotion_codes)
                        emotion_idx = code - 799;
                        emotion = emotion_labels{emotion_idx};
                        
                        % Look for corresponding intensity
                        intensity = NaN;
                        for Y = i+1:end_idx
                            intensity_code = SEGN.event(Y).type;
                            
                            if ismember(intensity_code, intensity_codes)
                                intensity = intensity_code - 900;
                                break;
                            end
                        end
                        
                        % Add emotion and intensity if not already present
                        if ~ismember(emotion, emotion_list)
                            emotion_list{end+1} = emotion;
                            intensity_list(end+1) = intensity;
                        end
                    end
                end
                
                % Identify the dominant emotion based on highest intensity
                [max_intensity, max_idx] = max(intensity_list);
                if ~isempty(max_idx)
                    dominant_emotion = emotion_list{max_idx};
                else
                    dominant_emotion = 'undefined';
                end
                
                % Extract the musical period based on code 788
                music_start_idx = find([SEGN.event(start_idx:end_idx).type] == 788, 1, 'first');
                if ~isempty(music_start_idx)
                    music_start_sample = SEGN.event(start_idx + music_start_idx - 1).latency;
                    
                    % Calculate the end of the musical period (12 seconds = 6000 samples)
                    music_end_sample = music_start_sample + 6000 - 1;
                    
                    % Ensure period doesn't exceed data limits
                    music_end_sample = min(music_end_sample, size(SEGN.data, 2));
                    
                    % Extract the segment data
                    SEGN_segment = pop_select(SEGN, 'point', [music_start_sample, music_end_sample]);
                else
                    SEGN_segment = [];
                end
                
                % Save results to file
                results_path = sprintf('%sRISULTATI\\sub-%d_task-run%d_segment%d.mat', base_path, z, j, seg_idx);
                save(results_path, 'emotion_list', 'intensity_list', 'dominant_emotion', 'SEGN_segment');
                fprintf('Complete results for segment %d saved in %s\n', seg_idx, results_path);
                
                % Display dominant emotion information
                fprintf('Segment %d: Dominant emotion = %s (Intensity = %d)\n', seg_idx, dominant_emotion, max_intensity);
            end
        end
        
        % Process events for run 1 - extract single segment at the end
        if j == 1
            % Check and convert event structure if needed
            if istable(SEGN.event)
                SEGN.event = table2struct(SEGN.event);
            end
            
            % Ensure latency field exists
            if ~isfield(SEGN.event, 'latency')
                for i = 1:length(SEGN.event)
                    if isfield(SEGN.event, 'onset')
                        SEGN.event(i).latency = SEGN.event(i).onset * SEGN.srate;
                    else
                        error('SEGN.event is missing the "latency" or "onset" field.');
                    end
                end
            end
            
            % Extract segment from end of recording
            total_samples_run1 = size(SEGN.data, 2);
            start_sample = total_samples_run1 - 10000;
            segment_length = round(12 * SEGN.srate);
            end_sample = min(start_sample + segment_length - 1, total_samples_run1);
            
            % Validate sample range
            if start_sample < 1 || end_sample > total_samples_run1
                error('Invalid sample range.');
            end
            
            % Extract the segment
            SEGN_segment = pop_select(SEGN, 'point', [start_sample, end_sample]);
            
            % Save the segment to file
            save_filename_segment = sprintf('%sRISULTATI\\sub-%d_task-run%d_segment1.mat', base_path, z, j);
            save(save_filename_segment, 'SEGN_segment');
            fprintf('Saved single segment in %s\n', save_filename_segment);
        end
        
        % FREQUENCY ANALYSIS SECTION
        % -------------------------
        % Define frequency bands
        freq_bands = struct(...
            'alpha', [8 12], ...
            'beta', [12 35], ...
            'delta', [0.5 4], ...
            'theta', [4 8], ...
            'gamma', [35 80], ...
            'smr', [12 15]);
        
        % Initialize variables for spectral analysis
        num_channels = size(SEGN.data, 1);
        nfft = 2^nextpow2(size(SEGN.icaact, 2));
        
        % Initialize power arrays for each frequency band
        alpha_power = zeros(num_channels, 1);
        beta_power = zeros(num_channels, 1);
        delta_power = zeros(num_channels, 1);
        theta_power = zeros(num_channels, 1);
        gamma_power = zeros(num_channels, 1);
        smr_power = zeros(num_channels, 1);
        
        % Initialize array for spectral indices
        I = zeros(num_channels, 37);
        
        % Process segments for run 2
        if j == 2
            for seg_idx = 1:num_segments
                % Load segment data
                filename = sprintf('%sRISULTATI\\sub-%d_task-run%d_segment%d.mat', base_path, z, j, seg_idx);
                load(filename, 'SEGN_segment');
                load(filename, 'dominant_emotion');
                
                % Initialize spectral index matrix
                num_channels = size(SEGN_segment.data, 1);
                I = zeros(num_channels, 37);
                nfft = 2^nextpow2(size(SEGN_segment.data, 2));
                
                % Process each channel
                for i = 1:num_channels
                    % Extract channel signal
                    signal = SEGN_segment.data(i, :);
                    
                    % Calculate power spectral density
                    [psd, freqs] = pwelch(signal, [], [], nfft, fs);
                    
                    % Find frequency band indices
                    alpha_idx = freqs >= freq_bands.alpha(1) & freqs <= freq_bands.alpha(2);
                    beta_idx = freqs >= freq_bands.beta(1) & freqs <= freq_bands.beta(2);
                    delta_idx = freqs >= freq_bands.delta(1) & freqs <= freq_bands.delta(2);
                    theta_idx = freqs >= freq_bands.theta(1) & freqs <= freq_bands.theta(2);
                    gamma_idx = freqs >= freq_bands.gamma(1) & freqs <= freq_bands.gamma(2);
                    smr_idx = freqs >= freq_bands.smr(1) & freqs <= freq_bands.smr(2);
                    
                    % Calculate band power values
                    alpha_power(i) = sum(psd(alpha_idx));
                    beta_power(i) = sum(psd(beta_idx));
                    delta_power(i) = sum(psd(delta_idx));
                    theta_power(i) = sum(psd(theta_idx));
                    gamma_power(i) = sum(psd(gamma_idx));
                    smr_power(i) = sum(psd(smr_idx));
                    
                    % Calculate spectral indices I1-I37
                    I(i, 1) = beta_power(i) / alpha_power(i); % I1
                    I(i, 2) = beta_power(i) / (alpha_power(i) + theta_power(i)); % I2
                    I(i, 3) = beta_power(i) / theta_power(i); % I3
                    I(i, 4) = theta_power(i) / alpha_power(i); % I4
                    I(i, 5) = theta_power(i) / delta_power(i); % I5
                    I(i, 6) = smr_power(i) / theta_power(i); % I6
                    I(i, 7) = smr_power(i) / beta_power(i); % I7
                    I(i, 8) = (alpha_power(i) + beta_power(i)) / delta_power(i); % I8
                    I(i, 9) = (theta_power(i) + alpha_power(i)) / (alpha_power(i) + beta_power(i)); % I9
                    I(i, 10) = theta_power(i) / (alpha_power(i) + beta_power(i)); % I10
                    I(i, 11) = (theta_power(i) + alpha_power(i)) / gamma_power(i); % I11
                    I(i, 12) = (beta_power(i) + theta_power(i)) / alpha_power(i); % I12
                    I(i, 13) = (delta_power(i) + theta_power(i)) / beta_power(i); % I13
                    I(i, 14) = (delta_power(i) + theta_power(i) + alpha_power(i)) / beta_power(i); % I14
                    I(i, 15) = (delta_power(i) + theta_power(i)) / alpha_power(i); % I15
                    I(i, 16) = (delta_power(i) + theta_power(i)) / (alpha_power(i) + beta_power(i)); % I16
                    I(i, 17) = delta_power(i) / alpha_power(i); % I17
                    I(i, 18) = delta_power(i) / beta_power(i); % I18
                    I(i, 19) = theta_power(i) / gamma_power(i); % I19
                    I(i, 20) = alpha_power(i) / gamma_power(i); % I20
                    I(i, 21) = (smr_power(i) + beta_power(i)) / theta_power(i); % I21
                    I(i, 22) = (theta_power(i) + alpha_power(i)) / (beta_power(i) + gamma_power(i)); % I22
                    I(i, 23) = (alpha_power(i) + beta_power(i)) / (alpha_power(i) + theta_power(i)); % I23
                    I(i, 24) = alpha_power(i) / (beta_power(i) + gamma_power(i)); % I24
                    I(i, 25) = (delta_power(i) + theta_power(i) + alpha_power(i)) / (beta_power(i) + gamma_power(i)); % I25
                    I(i, 26) = alpha_power(i) / (delta_power(i) + theta_power(i) + alpha_power(i)); % I26
                    I(i, 27) = alpha_power(i) / (theta_power(i) + alpha_power(i) + beta_power(i)); % I27
                    I(i, 28) = beta_power(i) / (theta_power(i) + gamma_power(i)); % I28
                    I(i, 29) = (beta_power(i) + gamma_power(i)) / delta_power(i); % I29
                    I(i, 30) = (alpha_power(i) + beta_power(i)) / gamma_power(i); % I30
                    I(i, 31) = (alpha_power(i) + gamma_power(i)) / (theta_power(i) + delta_power(i)); % I31
                    I(i, 32) = (theta_power(i) + alpha_power(i)) / delta_power(i); % I32
                    I(i, 33) = (theta_power(i) + beta_power(i)) / (alpha_power(i) + gamma_power(i)); % I33
                    I(i, 34) = (beta_power(i) + gamma_power(i)) / (delta_power(i) + theta_power(i)); % I34
                    I(i, 35) = (delta_power(i) + alpha_power(i)) / (theta_power(i) + gamma_power(i)); % I35
                    I(i, 36) = (theta_power(i) + alpha_power(i)) / (delta_power(i) + beta_power(i) + gamma_power(i)); % I36
                    I(i, 37) = (alpha_power(i) + beta_power(i)) / (delta_power(i) + theta_power(i) + gamma_power(i)); % I37
                end
                
                % Save results
                save_filename = sprintf('%sRISULTATI\\sub-%d_task-run%d_segment%d_results.mat', base_path, z, j, seg_idx);
                save(save_filename, 'I', 'SEGN_segment', 'dominant_emotion');
                fprintf('Complete results for segment %d saved in %s\n', seg_idx, save_filename);
            end
        end
        
        % Process segment for run 1
        if j == 1
            % Load segment data
            filename = sprintf('%sRISULTATI\\sub-%d_task-run%d_segment1.mat', base_path, z, j);
            load(filename, 'SEGN_segment');
            
            % Initialize indices matrix
            num_channels = size(SEGN_segment.data, 1);
            I = zeros(num_channels, 37);
            nfft = 2^nextpow2(size(SEGN_segment.data, 2));
            
            % Process each channel
            for i = 1:num_channels
                % Extract channel signal
                signal = SEGN_segment.data(i, :);
                
                % Calculate power spectral density
                [psd, freqs] = pwelch(signal, [], [], nfft, fs);
                
                % Find frequency band indices
                alpha_idx = freqs >= freq_bands.alpha(1) & freqs <= freq_bands.alpha(2);
                beta_idx = freqs >= freq_bands.beta(1) & freqs <= freq_bands.beta(2);
                delta_idx = freqs >= freq_bands.delta(1) & freqs <= freq_bands.delta(2);
                theta_idx = freqs >= freq_bands.theta(1) & freqs <= freq_bands.theta(2);
                gamma_idx = freqs >= freq_bands.gamma(1) & freqs <= freq_bands.gamma(2);
                smr_idx = freqs >= freq_bands.smr(1) & freqs <= freq_bands.smr(2);
                
                % Calculate band power values
                alpha_power(i) = sum(psd(alpha_idx));
                beta_power(i) = sum(psd(beta_idx));
                delta_power(i) = sum(psd(delta_idx));
                theta_power(i) = sum(psd(theta_idx));
                gamma_power(i) = sum(psd(gamma_idx));
                smr_power(i) = sum(psd(smr_idx));
                
                % Calulate indexes I1 to I37
                I(i, 1) = beta_power(i) / alpha_power(i); % I1
                I(i, 2) = beta_power(i) / (alpha_power(i) + theta_power(i)); % I2
                I(i, 3) = beta_power(i) / theta_power(i); % I3
                I(i, 4) = theta_power(i) / alpha_power(i); % I4
                I(i, 5) = theta_power(i) / delta_power(i); % I5
                I(i, 6) = smr_power(i) / theta_power(i); % I6
                I(i, 7) = smr_power(i) / beta_power(i); % I7
                I(i, 8) = (alpha_power(i) + beta_power(i)) / delta_power(i); % I8
                I(i, 9) = (theta_power(i) + alpha_power(i)) / (alpha_power(i) + beta_power(i)); % I9
                I(i, 10) = theta_power(i) / (alpha_power(i) + beta_power(i)); % I10
                I(i, 11) = (theta_power(i) + alpha_power(i)) / gamma_power(i); % I11
                I(i, 12) = (beta_power(i) + theta_power(i)) / alpha_power(i); % I12
                I(i, 13) = (delta_power(i) + theta_power(i)) / beta_power(i); % I13
                I(i, 14) = (delta_power(i) + theta_power(i) + alpha_power(i)) / beta_power(i); % I14
                I(i, 15) = (delta_power(i) + theta_power(i)) / alpha_power(i); % I15
                I(i, 16) = (delta_power(i) + theta_power(i)) / (alpha_power(i) + beta_power(i)); % I16
                I(i, 17) = delta_power(i) / alpha_power(i); % I17
                I(i, 18) = delta_power(i) / beta_power(i); % I18
                I(i, 19) = theta_power(i) / gamma_power(i); % I19
                I(i, 20) = alpha_power(i) / gamma_power(i); % I20
                I(i, 21) = (smr_power(i) + beta_power(i)) / theta_power(i); % I21
                I(i, 22) = (theta_power(i) + alpha_power(i)) / (beta_power(i) + gamma_power(i)); % I22
                I(i, 23) = (alpha_power(i) + beta_power(i)) / (alpha_power(i) + theta_power(i)); % I23
                I(i, 24) = alpha_power(i) / (beta_power(i) + gamma_power(i)); % I24
                I(i, 25) = (delta_power(i) + theta_power(i) + alpha_power(i)) / (beta_power(i) + gamma_power(i)); % I25
                I(i, 26) = alpha_power(i) / (delta_power(i) + theta_power(i) + alpha_power(i)); % I26
                I(i, 27) = alpha_power(i) / (theta_power(i) + alpha_power(i) + beta_power(i)); % I27
                I(i, 28) = beta_power(i) / (theta_power(i) + gamma_power(i)); % I28
                I(i, 29) = (beta_power(i) + gamma_power(i)) / delta_power(i); % I29
                I(i, 30) = (alpha_power(i) + beta_power(i)) / gamma_power(i); % I30
                I(i, 31) = (alpha_power(i) + gamma_power(i)) / (theta_power(i) + delta_power(i)); % I31
                I(i, 32) = (theta_power(i) + alpha_power(i)) / delta_power(i); % I32
                I(i, 33) = (theta_power(i) + beta_power(i)) / (alpha_power(i) + gamma_power(i)); % I33
                I(i, 34) = (beta_power(i) + gamma_power(i)) / (delta_power(i) + theta_power(i)); % I34
                I(i, 35) = (delta_power(i) + alpha_power(i)) / (theta_power(i) + gamma_power(i)); % I35
                I(i, 36) = (theta_power(i) + alpha_power(i)) / (delta_power(i) + beta_power(i) + gamma_power(i)); % I36
                I(i, 37) = (alpha_power(i) + beta_power(i)) / (delta_power(i) + theta_power(i) + gamma_power(i)); % I37
            end
            
            % Salva il file dei risultati
            save_filename = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run%d_segment1_results.mat', z, j);
            save(save_filename, 'I', 'SEGN_segment');
            fprintf('Risultati completi salvati in %s\n', save_filename);
            end
    end
end



% Initialize data structures for results
p_values_all = zeros(19, 37, 8, 7); % P-values matrix (channels, indices, main emotion, comparison emotion)
h_values_all = zeros(19, 37, 8, 7); % Hypothesis test results matrix

% Define emotion labels
emotion_labels = {'happy', 'sad', 'tender', 'angry', 'tense', 'pleasant', 'afraid', 'energetic'};

% Process each emotion as the main emotion
for emotion_idx = 1:length(emotion_labels) % Main emotion loop
    % Process each data channel
    for channel = 1:19 % Channel loop
        for index_pos = 1:37 % Index position loop
            % Create data containers
            main_emotion_data = [];
            other_emotions_data = cell(1, 7); % Container for other emotions
            
            % Process each subject's data
            for subject_id = 12:21 % Loop through subject IDs
                % Process segments for each subject
                for segment_num = 1:10 % Process 10 segments per subject
                    % Construct path to data file
                    data_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\RISULTATI\\sub-%d_task-run2_segment%d_results.mat', subject_id, segment_num);
                    
                    % Attempt to load the data file
                    try
                        loaded_data = load(data_path, 'I', 'dominant_emotion');
                        I = loaded_data.I;
                        dominant_emotion = loaded_data.dominant_emotion;
                    catch exception
                        warning('Unable to load file: %s', data_path);
                        continue; % Skip to next iteration
                    end
                    
                    % Extract the value at current channel and index
                    current_value = I(channel, index_pos);
                    
                    % Sort value based on detected emotion
                    if strcmp(dominant_emotion, emotion_labels{emotion_idx})
                        % Add to main emotion data
                        main_emotion_data = [main_emotion_data, current_value];
                    else
                        % Add to appropriate comparison emotion
                        for comp_idx = 1:length(emotion_labels)
                            if comp_idx ~= emotion_idx && strcmp(dominant_emotion, emotion_labels{comp_idx})
                                % Adjust index when storing in the cell array
                                adjusted_idx = comp_idx - (comp_idx > emotion_idx);
                                other_emotions_data{adjusted_idx} = [other_emotions_data{adjusted_idx}, current_value];
                            end
                        end
                    end
                end
            end
            
            % Perform statistical tests for each emotion comparison
            for comp_idx = 1:length(emotion_labels)-1
                % Check if we have sufficient data for comparison
                if ~isempty(main_emotion_data) && ~isempty(other_emotions_data{comp_idx})
                    % Ensure equal length for paired test
                    sample_size = min(length(main_emotion_data), length(other_emotions_data{comp_idx}));
                    main_samples = main_emotion_data(1:sample_size);
                    comparison_samples = other_emotions_data{comp_idx}(1:sample_size);
                    
                    % Perform Wilcoxon signed rank test
                    [p_value, h_result] = signrank(main_samples, comparison_samples);
                else
                    % No comparison possible
                    p_value = NaN;
                    h_result = 0;
                end
                
                % Store results
                p_values_all(channel, index_pos, emotion_idx, comp_idx) = p_value;
                h_values_all(channel, index_pos, emotion_idx, comp_idx) = h_result;
            end
        end
    end
end

% Visualize and save results
for emotion_idx = 1:length(emotion_labels)
    % Create new figure for current emotion
    figure; 
    sgtitle(['Comparisons for: ', emotion_labels{emotion_idx}]);
    
    subplot_counter = 1;
    for comp_idx = 1:length(emotion_labels)-1
        % Calculate the actual comparison emotion index
        actual_comp_idx = comp_idx + (comp_idx >= emotion_idx);
        
        % Create subplot
        subplot(2, 4, subplot_counter);
        
        % Get current comparison data
        current_p_values = p_values_all(:, :, emotion_idx, comp_idx);
        current_h_values = h_values_all(:, :, emotion_idx, comp_idx);
        
        % Visualize p-values
        imagesc(current_p_values);
        colorbar; 
        caxis([0.01 0.1]);
        title(['P-values: ', emotion_labels{emotion_idx}, ' vs ', emotion_labels{actual_comp_idx}]);
        xlabel('Indices'); 
        ylabel('Channels');
        
        % Increment subplot counter
        subplot_counter = subplot_counter + 1;
        
        % Save matrices to file
        result_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\MATRICI\\Pvalues_%s_vs_%s.mat', emotion_labels{emotion_idx}, emotion_labels{actual_comp_idx});
        save(result_path, 'current_p_values', 'current_h_values');
    end
    
    % Save the figure
    figure_path = sprintf('C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\MATRICI\\Confronti_%s.png', emotion_labels{emotion_idx});
    saveas(gcf, figure_path);
end