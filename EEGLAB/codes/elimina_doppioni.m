folder = 'C:\\Users\\giuseppe\\OneDrive\\Desktop\\magistrale biomedical engineering engineering of medical devices\\brain research\\EEGLAB\\all_emotions_vs_others_file_txt'; 
cd(folder);

% Prendi tutti i file che iniziano con 'frequenza canale' o 'frequenza indice'
files = dir(fullfile(folder, 'frequenza vs.txt'));

% Mappa per salvare solo una combinazione per ciascuna coppia
unique_pairs = containers.Map;

for i = 1:length(files)
    name = files(i).name;

    % Estrai le due emozioni ignorando maiuscole, spazi e ordine
    parts = regexp(lower(name), 'frequenza (?:canale|indice)[ _](\w+)[ _]?vs[ _]?(\w+)', 'tokens');

    if ~isempty(parts)
        emotions = sort(parts{1}); % ordina alfabeticamente
        key = strjoin(emotions, '_'); % chiave standard, es: afraid_angry

        if ~isKey(unique_pairs, key)
            unique_pairs(key) = name; % conserva questo file
        else
            % Duplicato trovato, lo elimino
            delete(fullfile(folder, name));
            fprintf('File eliminato: %s\n', name);
        end
    end
end

fprintf('Pulizia completata. Sono stati mantenuti %d file unici.\n', unique_pairs.Count);