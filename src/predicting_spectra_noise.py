import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
from Dataset_generator import process_combination
from typing import List, Tuple

labels = np.loadtxt('/home/beck/star-formation-ML/data/parameterset_plusMass_smallRadius.txt')
log_runs_dir = '/home/beck/star-formation-ML/logs/eval/runs/'

# Listet den Inhalt des Predictions-Verzeichnisses auf
log_runs = os.listdir(log_runs_dir)

# Sortiert die Versionen nach dem Erstellungsdatum
sorted_runs = sorted(log_runs, key=lambda x: os.path.getmtime(os.path.join(log_runs_dir, x)))

# Die neueste Version ist die letzte in der sortierten Liste
aktuelle_version = sorted_runs[-2]

y_pred = np.load('%s%s/y_pred.npy' %(log_runs_dir, aktuelle_version))
y_pred = np.append(y_pred, np.tile(labels[0,5:], (len(y_pred), 1)), axis=1)

# Do I want to generate all spectra of the predicted samples or just a little sample?
random_sample = 'no' # yes or no for a random sample with number of samples defined below
number_samples = 10

random_numbers = np.random.randint(0, len(y_pred), size=number_samples)
indices = random_numbers.astype(int)  # Konvertieren der Indizes in Ganzzahlen


# Erzeugen aller Kombinationen der Parameter
if random_sample == 'yes':
    combinations = y_pred[indices, :6]
else:
    combinations = y_pred[:,:6]

x_pred = np.empty((0, 4302))  # this is the array for the finished dataset, which is empty at the beginning

# Anzahl der Prozesse festlegen (z.B. Anzahl der verfügbaren CPU-Kerne)
num_processes = 60

results = []
with Pool(num_processes) as pool:
    try:
        for result in tqdm(pool.imap(process_combination, combinations), total=len(combinations)):
            results.append(result)
    except TypeError:
        # Fehler ignorieren und mit dem Code fortfahren
        pass
    pool.close()
    pool.join()
    
for Spectrum_array, parameter_array, Frequency_array in results:
    x_pred = np.vstack((x_pred, Spectrum_array))
    Frequency_array  = Frequency_array


# Speichern der Daten

np.save('%s%s/x_pred.npy' %(log_runs_dir, aktuelle_version), x_pred)
np.save('%s%s/Frequency_array.npy' %(log_runs_dir, aktuelle_version), Frequency_array)