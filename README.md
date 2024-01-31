# EEG_Sintetic_MorphoGenerator

# Base de Datos de Señales EEG - DataBaseV1

La base de datos `DataBaseV1` contiene una colección de señales EEG centradas en eventos con morfologías punta, lenta y punta-onda lenta. Esta base de datos está organizada en las carpetas `Spike_Waves`, `Slow_Waves` y`Spike_Slow_Waves`, donde se clasifican las señales según la naturaleza y la localización de los eventos EEG.

## Estructura de Carpetas

Dentro de `Spike_Slow_Waves`, las señales están distribuidas en subcarpetas que representan diferentes tipos de eventos:

- `Complex`: Contiene eventos complejos, caracterizados por agrupaciones o trenes de ondas de más de 4 segundos que son claramente identificables del fondo EEG. Esta categoría se divide a su vez en:
    - `Middle`: Eventos complejos localizados en el medio de la señal.
    - `Random`: Eventos complejos presentes de forma aleatoria en la señal.
- `Transient`: Incluye eventos transitorios, que corresponden a ondas identificables de la señal de fondo de manera aislada.

## Nomenclatura de los Archivos

Cada archivo en la base de datos sigue una nomenclatura específica que describe las características de la señal:

- `Sp`: Spike Signal (Señal Punta).
- `Sl`: Slow Signal (Señal Lenta).
- `Clpx`: Complex Event (Evento Complejo).
    - `Midd`: Complex Event Middle (Evento Complejo en Medio).
    - `Rand`: Complex Event Random (Evento Complejo Aleatorio).
- `Trnt`: Transient Event (Evento Transitorio).
- `NE`: Noise EEG (Ruido del EEG).
- `NW`: Noise Wave (Ruido de las Ondas).

Esta nomenclatura permite una rápida identificación de las características principales de cada señal dentro de la base de datos.






<p align="center">
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
</p>
