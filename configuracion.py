class configuracion:

    def __init__(self):
        """
            Constructor de la clase de configuración.
        """
        self.numEtiquetas = 0 # Entero con el número de tipos de denuncias que hay.
        self.carpetasEtiquetas = [] # Vector para almacenar el listando de la carpeta Corpus.
        self.carpetaCorpus = "" # String con la ruta de la carpeta en la que se encuentra el corpus.
        self.modeloPreentrenado= "" # String con el modelo a usar en el entrenamiento.
        self.archivoParametros= "" # String con el nombre del archivo del parámetros.
        self.rutas = {} # Vector para guardar las rutas de cada tipo de denuncia.
        self.comentario="" # String con el comentario identificativo de la ejeución.
        self.k=0 # Entero con el número de conjuntos a dividir e iteraciones en la validación k-fold.
        self.numDenunciasPorTipo=0 # Entero con las denuncias a procesar para ambos tipos.
        self.modo="" # String con el modo de ejecución. Entrenamiento o Validación.
        self.procesamiento=0 # Entero con el tipo de procesamiento de los textos.
        self.semilla=0 # Entero con la semilla de la ejecución.

    def lecturaParametros(self):
        """
            Función para abrir el archivo parámetros.txt y guardar su información.
        """
        import os
        print("---------------------Cargando archivo de parámetros-----------------------")
        self.archivoParametros = open('Parámetros.txt', 'r')
        for linea in self.archivoParametros:
            contenido = linea.split("=")  # Dividimos cada línea en dos partes (antes y detrás del espacio).
            if contenido[0] == "carpetaCorpus": # Identificamos cada primera parte.
                try:
                    self.carpetaCorpus = contenido[1].strip()  # Quitamos espacios innecesarios, si hubiera.
                    self.carpetasEtiquetas = os.listdir(self.carpetaCorpus) # Almacenamos cada tipo de etiqueta.
                    self.numEtiquetas=len(self.carpetasEtiquetas) # Almacenamos el número de archivos.

                    for etiqueta in self.carpetasEtiquetas:
                        self.rutas[etiqueta]=self.carpetaCorpus+"\\"+etiqueta # Creamos la ruta de cada tipo de denuncia.
                except FileNotFoundError:
                    print("Error, carpeta no detectada")
            if contenido[0] == "modeloPreentrenado":
                self.modeloPreentrenado=contenido[1].strip()

            if contenido[0] == "comentario":
                self.comentario=contenido[1].strip()

            if contenido[0] == "k":
                self.k = int(contenido[1].strip())

            if contenido[0] == "numDenunciasPorTipo":
                self.numDenunciasPorTipo = int(contenido[1].strip())

            if contenido[0] == "modo":
                self.modo = contenido[1].strip()

            if contenido[0] == "procesamiento":
                self.procesamiento = int(contenido[1].strip())

            if contenido[0] == "semilla":
                self.semilla = int(contenido[1].strip())

        self.mostrarConfiguracion()
        self.archivoParametros.close()

    def mostrarConfiguracion(self):
        """
            Función para mostrar la información del fichero de configuración.
        """
        print("Etiquetas: ",self.getEtiquetas)
        print("Modelo Preentrenado: ",self.modeloPreentrenado)
        print("Rutas Entrenamiento: ",self.rutas)
        print()

    @property
    def getNumEtiquetas(self):
        """
            Getter del número de etiquetas.
            Devuelve:
                - Entero con el número de etiquetas.
        """
        return self.numEtiquetas

    @property
    def getModeloPreentrenado(self):
        """
            Getter del modelo a usar en el entrenamiento.
            Devuelve:
                - String con el modelo.
        """
        return self.modeloPreentrenado

    @property
    def getComentario(self):
        """
            Getter del comentario indentificativo de la ejecución.
            Devuelve:
                - String con el comentario
        """
        return self.comentario

    @property
    def getEtiquetas(self):
        """
            Getter del las etiquetas disponibles para la clasificación.
            Devuelve:
                - Lista con los nombres de las diferentes carpetas de etiquetas.
        """
        return self.carpetasEtiquetas

    @property
    def getRutas(self):
        """
            Getter del las rutas de las diferentes carpetas de textos según etiqueta.
            Devuelve:
                - Lista con las rutas de cada tipo de denuncia.
        """
        return self.rutas

    @property
    def getCarpetaCorpus(self):
        """
            Getter de la carpeta en la que se encuentra el corpus.
            Devuelve:
                - String con la carpeta del corpus.
        """
        return self.carpetaCorpus

    @property
    def getNumDenunciasPorTipo(self):
        """
            Getter del número de denuncias a procesar por tipo.
            Devuelve:
                - Entero con el número de denuncias.
        """
        return self.numDenunciasPorTipo

    @property
    def getK(self):
        """
            Getter del número de conjuntos a dividir e iteraciones en la validación k-fold.
            Devuelve:
                - Entero con el k a usar.
        """
        return self.k

    @property
    def getModo(self):
        """
            Getter del modo de ejecución.
            Devuelve:
                - String con el modo de ejecución actual.
        """
        return self.modo

    @property
    def getSemilla(self):
        """
            Getter del número de la semilla de la ejecución.
            Devuelve:
                - Entero con la semilla a usar.
        """
        return self.semilla

    @property
    def getProcesamiento(self):
        """
            Getter del procesamiento elegido.
            Devuelve:
                - Entero con el identificador del procesamiento.
        """
        return self.procesamiento

