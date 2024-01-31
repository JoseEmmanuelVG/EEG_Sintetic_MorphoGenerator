import os

def rename_files(directory):
    for filename in os.listdir(directory):
        if filename.startswith("Sp_Sl_Cplx_Midd_NE100_NW1_JEVG"):
            # Divide el nombre del archivo en partes y verifica si tiene un número al final
            parts = filename.split("_")
            if parts[-1].startswith("JEVG"):
                number = parts[-1][4:]  # Extrae el número después de "JEVG"
                new_name = "Sp_Sl_Trnt_NE100_NW1_JEVG" + number
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_name)
                os.rename(old_file, new_file)
                print(f"Renombrado: {filename} a {new_name}")
cd Spike_Slow_Waves/Transient

# Ruta de la carpeta que contiene los archivos
directory = r"/workspaces/EEG_Sintetic_MorphoGenerator/Spike_Slow_Waves/Transient"

rename_files(directory)
