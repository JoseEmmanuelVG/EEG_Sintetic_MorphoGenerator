import mne

# Cargamos el archivo EDF con MNE
raw = mne.io.read_raw_edf("output_file.edf", preload=True)

# Mostrar los datos en una gráfica (esto se abrirá en una ventana)
fig = raw.plot(n_channels=15, scalings={"eeg": 100e-6}, show=False)  # Ajusta n_channels y scalings según tus necesidades

# Guardar la gráfica como una imagen
fig.savefig('EEG_plot.png')
