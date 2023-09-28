import configuracion

class lecturaArchivos:
    def __init__(self,configuracion):
        """
            Constructor de la clase.
            Parámetros:
                - configuracion: Instancia de la clase configuración con los parámetros de ejecución.
        """
        self.configuracion=configuracion # Información de configuración de la ejecución.
        self.mapaDenuncias = {} # Mapa para almacenar las denuncias según su tipo.

    def cargar(self):
        """
            Función  para la lectura automatizada de todos los archivos .docx del establecido en el fichero de parámetros.
        """
        import os  # Abrimos la carpeta.
        import docx2txt
        import glob  # Lectura y procesado de los archivos.

        print("---------------------Leyendo Corpus-----------------------")
        os.chdir(self.configuracion.getCarpetaCorpus)

        for carpeta in os.listdir():
            print("Leyendo",carpeta)
            os.chdir(carpeta)  # Recorremos cada directorio y leemos los archivos.
            archivosProcesados = [] #Mapa de listas de textos por carpeta.
            self.mapaDenuncias[carpeta] = archivosProcesados
            i=0
            for filename in glob.glob('*.docx'):  # Lectura de todos los archivos de la carpeta con la libreria glob.
                with open(os.path.join(os.getcwd(), filename), 'r') as f:
                    text = docx2txt.process(filename) # Convertimos a formato texto el docx.
                    if(text!=""): # Si no está vacío.
                        print(i,carpeta, "/", filename, " Leído")
                        i=i+1
                    self.mapaDenuncias[carpeta].append(text) # Los guardamos.
            os.chdir('..')  # Volvemos al directorio padre.
        os.chdir('..')
    @property
    def getArchivosODIO(self):
        """
            Getter de la lista de denuncias de ODIO.
            Devuelve:
                - Lista con las denuncias de ODIO.
        """
        return self.mapaDenuncias["ODIO"]

    @property
    def getArchivosNOODIO(self):
        """
            Getter de la lista de denuncias de NO ODIO.
            Devuelve:
                - Lista con las denuncias de NO ODIO.
        """
        return self.mapaDenuncias["NOODIO"]

    @property
    def muestraArchivosODIO(self):
        """
            Función para mostrar la lista de denuncias de ODIO.
        """
        for elemento in self.mapaDenuncias["ODIO"]:
            print(elemento)

    @property
    def muestraArchivosNOODIO(self):
        """
            Función para mostrar la lista de denuncias de NO ODIO.
        """
        for elemento in self.mapaDenuncias["NOODIO"]:
            print(elemento)

