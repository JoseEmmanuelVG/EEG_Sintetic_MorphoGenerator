# Generador de EEG Sintético con Eventos Epileptogénicos

## Descripción
Herramienta para generar señales electroencefalográficas (EEG) sintéticas, especialmente enfocada en morfologías (grafo elementos), eventos y actividad de fondo relacionados con la epilepsia.

## Versión de Software y Compatibilidad
El desarrollo y las pruebas del presente proyecto se llevaron a cabo utilizando las siguientes versiones de software:

- Python: Versión 3.9.13
- Dash: Versión 2.11.1
- Visual Studio Code: 1.86.2

Se recomienda verificar y utilizar las versiones mencionadas para asegurar compatibilidad y funcionamiento adecuado.

## Instalación del Entorno y Dependencias

### Preparación del Entorno
- **Editor de Código:** Se utiliza Visual Studio Code (VSCode) como editor de código fuente, aunque se puede seleccionar el de preferencia.
- **Acceso al Código:** Abre la carpeta del proyecto en VSCode, asegurándose de incluir todos los archivos y subcarpetas.

### Configuración del Entorno Python
- Si no tiene Python instalado, descárguelo e instálelo desde el sitio web oficial [python.org](https://www.python.org/).
- Asegúrate de agregar Python a las variables de entorno del sistema.

### Instalación de Dependencias
Ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install dash matplotlib plotly numpy scipy pyedflib mne waitress kaleido -U
```


## Ejecución de la Aplicación

### Utilización del Servidor Waitress

- Abre la terminal integrada en VSCode o cualquier terminal de tu elección.
- Navega hasta la ubicación de la carpeta del proyecto y ejecuta el siguiente comando para iniciar la aplicación:

```bash
waitress-serve --call 'main:app'
```

### Acceso a la Aplicación

- Una vez iniciada la aplicación, se mostrará en la terminal el puerto local utilizado (por ejemplo, http://0.0.0.0:8080).
- Abre un navegador web y accede a la dirección indicada para visualizar la interfaz principal de la aplicación.

## Descripción del código
Para obtener más detalles sobre la instalación, configuración y uso del software, consulta el archivo "Codigo_Fuente_JEVG" incluido en el proyecto.



<p align="center">
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
<a rel="license" href="https://github.com/JoseEmmanuelVG"></a><br />By<a rel="license" href="https://github.com/JoseEmmanuelVG">   J.E.V.G</a>.
</p>
