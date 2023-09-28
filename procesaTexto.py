from transformers import DistilBertTokenizer


class procesaTexto:
    def __init__(self, delitosodio, delitosnodio):
        """
            Constructor de la clase.
            Parámetros:
                - delitosodio: Lista con todas las denuncias de Odio
                - delitosnodio: Lista con todas las denuncuas de No Odio
        """
        self.delitosodio = delitosodio # Vector con las denuncias de Odio
        self.delitosnodio = delitosnodio # Vector con las denuncias de No Odio

    def procesamiento(self):
        """
            Procesamiento que elimina espacios vacíos irrelevantes del formateo .docx.
        """
        print("Procesando")
        i = 0
        for delito in self.delitosodio: # Por cada delito de odio extraemos todas las palabras.
            palabras = delito.split()
            listaAlfa = []
            for palabra in palabras:
                listaAlfa.append(palabra)
            textoAlfa = ' '.join(listaAlfa) # Creamos una cadena con todas las palabras extraidas separadas por un espacio.
            self.delitosodio[i] = textoAlfa
            i = i + 1
        i = 0
        for delito in self.delitosnodio: # Por cada delito de no odio extraemos todas las palabras.
            palabras = delito.split()
            # print(texto.split())
            listaAlfa = []
            for palabra in palabras:
                listaAlfa.append(palabra)
            textoAlfa = ' '.join(listaAlfa) # Creamos una cadena con todas las palabras extraidas separadas por un espacio.

            self.delitosnodio[i] = textoAlfa # Almacenamos el resultado por referencia
            i = i + 1

    def procesamiento2(self):
        """
            Descripción del procesamiento:
                - Elimina espacios vacíos irrelevantes del formateo .docx.
                - No se tienen en cuenta los carácteres no alfabéticos:
                    - Cabecera con el nombre del documento.
                    - Etiquetas de anonimizado.
                - Elimina los signos de puntuación.
        """
        import string
        # print(string.punctuation)
        print("Procesando")
        i = 0
        for delito in self.delitosodio: # Denuncias de Odio
            lista = []
            for letra in delito: # Recorremos todos los caracteres de cada delito de odio.
                if letra not in string.punctuation: # Nos quedamos con las que no son signos de puntuación.
                    lista.append(letra)
            texto = ''.join(lista) # Los almacenamos en una cadena de texto, sin separación por espacio.
            palabras = texto.split() # Extraemos todas las palabras de esa cadena de texto.
            # print(texto.split())
            listaAlfa = []
            for palabra in palabras:
                # print(palabra.isalpha(),palabra)
                if not palabra.isupper(): # Nos quedamos con las palabras que no están completamente en mayúscula
                    if palabra.isalpha() or palabra.isnumeric(): # Nos quedamos con las palabras que son alfabéticas o numéricas
                        listaAlfa.append(palabra)
            textoAlfa = ' '.join(listaAlfa) # Convertimos la lista resultante a una cadena de texto con cada palabra separada por espacios.

            self.delitosodio[i] = textoAlfa # Almacenamos el resultado por referencia
            i = i + 1
        i=0
        for delito in self.delitosnodio: # Por cada delito de no odio extraemos todas las palabras
            lista = []
            for letra in delito:
                # print(letra)
                if letra not in string.punctuation: # Nos quedamos con las que no son signos de puntuación.
                    lista.append(letra)
            texto = ''.join(lista) # Los almacenamos en una cadena de texto, sin separación por espacio.
            palabras = texto.split() # Extraemos todas las palabras de esa cadena de texto.
            # print(texto.split())
            listaAlfa = []
            for palabra in palabras:
                # print(palabra.isalpha(),palabra)
                if not palabra.isupper(): # Nos quedamos con las palabras que no están completamente en mayúscula
                    if palabra.isalpha() or palabra.isnumeric(): # Nos quedamos con las palabras que son alfabéticas o numéricas
                        listaAlfa.append(palabra)
            textoAlfa = ' '.join(listaAlfa) # Convertimos la lista resultante a una cadena de texto con cada palabra separada por espacios.


            self.delitosnodio[i] = textoAlfa # Almacenamos el resultado por referencia
            i = i + 1
    def procesamiento3(self):
        """
            Descripción del procesamiento:
                      - Elimina espacios vacíos irrelevantes del formateo .docx.
                      - No se tienen en cuenta los carácteres no alfabéticos:
                          - Cabecera con el nombre del documento.
                          - Etiquetas de anonimizado.
                      - Elimina los signos de puntuación.
                      - Elimina frase común al tipo de denuncia:
                          - ODIO: No tiene en cuenta las palabras desde “Instructor” hasta “ocurridos”
                          - NO ODIO: No tiene en cuenta las palabras desde “Atestado” hasta “Dependencia”
        """
        import string
        # print(string.punctuation)
        print("procesando")
        i = 0
        for delito in self.delitosodio: # Denuncias de Odio
            lista = []
            for letra in delito:
                # print(letra)
                if letra not in string.punctuation: # Nos quedamos con los carácteres que no son signos de puntuación
                    lista.append(letra)
            texto = ''.join(lista) # Los almacenamos en una cadena de texto, sin separación por espacio.
            palabras = texto.split() # Extraemos las palabras con un split()
            # print(texto.split())
            listaAlfa = []
            for palabra in palabras:
                # print(palabra.isalpha(),palabra)
                if not palabra.isupper(): # Nos quedamos con las palabras que no están completamente en mayuscula
                    if palabra.isalpha() or palabra.isnumeric() and not palabra.isupper(): # Nos quedamos con las palabras alfabéticas y numéricas
                        listaAlfa.append(palabra)
            textoAlfa = ' '.join(listaAlfa)  # Almacenamos cada palabra separada por un espacio en una cadena de texto
            import re
            patron = r"Instructor\s(.*?)\socurridos" # Eliminamos las palabras entre Instructor y ocurridos en las denuncias de Odio
            texto_modificado = re.sub(patron, '', textoAlfa)
            # print(textoAlfa.split())
            self.delitosodio[i] = texto_modificado # Almacenamos el resultado por referencia
            i = i + 1
        i = 0
        for delito in self.delitosnodio: # Denuncias de No Odio
            lista = []
            for letra in delito:
                # print(letra)
                if letra not in string.punctuation: # Nos quedamos con los carácteres que no son signos de puntuación
                    lista.append(letra)
            texto = ''.join(lista)
            palabras = texto.split() #Almacenamos los caracteres en una cadena de texto
            # print(texto.split())
            listaAlfa = []
            for palabra in palabras:
                # print(palabra.isalpha(),palabra)
                if not palabra.isupper():  # Nos quedamos con las palabras que no son mayúsculas completamente
                    if palabra.isalpha() or palabra.isnumeric(): # Nos quedamos con las palabras alfabéticas y numéricas
                        listaAlfa.append(palabra)
            textoAlfa = ' '.join(listaAlfa) # Creamos una cadena de texto con las palabras separadas por espacios
            # print(textoAlfa.split())
            import re
            patron = r"Atestado\s(.*?)\Dependencia"  # Eliminamos las palabras entre Atestado y Dependencia en las denuncias de No Odio
            texto_modificado = re.sub(patron, '', textoAlfa)
            # print(textoAlfa.split())

            self.delitosnodio[i] = texto_modificado # Almacenamos el resultado por referencia
            i = i + 1

    def procesamiento4CalcularPalabras(self, nComunes, tenerEnCuentaStopWords):
        """
            Función para calcular la bolsa de palabras a quitar en el procesamiento 4. Es decir:
                - Palabras comunes de cada clase, es decir, palabras presentes en todos los documentos de una misma clase.
                - Palabras más repetidas de cada clase.
                - Palabras exclusivas de cada clase, es decir, palabras que aparecen en todos los documentos de una misma clase y no aparecen en el otro tipo de denuncia.
            Parámetros:
                - nComunes: Entero con el número de palabras más repetidas a tener en cuenta
                - tenerEnCuentaStopWords: Booleano que indica si tenenemos en cuenta a las stopwords para el borrado

        """
        # Cálculo de palabras que aparecen en todas las denuncias de No Odio
        from collections import Counter
        self.palabrasComunesNOODIO=self.delitosnodio[0].split()
        for noodio in self.delitosnodio: # Se hace una intersección de palabras presentes en la bolsa y en la siguiente denuncia
            self.palabrasComunesNOODIO=set(noodio.lower().split()).intersection(set(self.palabrasComunesN0ODIO)) # Iterativamente calculamos las palabras que concurren en todos los documentos de este tipo
        print("")
        print("Palabras comunes en ODIO", self.palabrasComunesODIO)
        # Cálculo de palabras que aparecen en todas las denuncias de Odio
        self.palabrasComunesODIO = self.delitosodio[0].lower().split()
        for odio in self.delitosodio: # Se hace una intersección de palabras presentes en la bolsa y en la siguiente denuncia
            self.palabrasComunesODIO=set(odio.lower().split()).intersection(set(self.palabrasComunesODIO))  # Iterativamente calculamos las palabras que concurren en todos los documentos de este tipo
        print("Palabras comunes en NO ODIO", self.palabrasComunesODIO)
        print("")

        # Cálculo de palabras exclusivas de cada clase
        self.exclusivasODIO= self.palabrasComunesODIO - self.palabrasComunesNOODIO # Palabras que sólo aparecen en todas las denuncias de No Odio
        self.exclusivasNOODIO= self.palabrasComunesNOODIO - self.palabrasComunesODIO # Palabras que sólo aparecen en todas las denuncias de Odio
        print("Exclusivas ODIO", self.exclusivasODIO)
        print("Exclusivas NO ODIO", self.exclusivasNOODIO)

        # Cálculo de las palabras que mas se repiten en cada tipo de denuncia
        textoGeneralOdio = ""
        textoGeneralNOOdio = ""
        for odio in self.delitosodio:
            textoGeneralOdio=textoGeneralOdio+odio # Creamos un texto con todas las denuncias de No Odio
        for noodio in self.delitosnodio:
            textoGeneralNOOdio=textoGeneralNOOdio+noodio # Creamos un texto con todas las denuncias de Odio
        print("")
        contadorNOOdio=Counter(textoGeneralNOOdio.lower().split()) # Contador de palabras que más se repiten en denuncias de No Odio
        contadorOdio=Counter(textoGeneralOdio.lower().split()) # Contador de palabras que más se repiten en denuncias de Odio

        import nltk
        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stopwords=stopwords.words('spanish')

        self.masRepetidasODIO=[]
        for palabra in contadorOdio.most_common(nComunes):
            if tenerEnCuentaStopWords==False: # Si no estamos contando con las stopwords para el conteo
                if palabra[0] not in stopwords:
                    self.masRepetidasODIO.append(palabra)
            else:
                self.masRepetidasODIO.append(palabra)

        self.masRepetidasNOODIO = []
        for palabra in contadorNOOdio.most_common(nComunes):
            if tenerEnCuentaStopWords==False:  # Si no estamos contando con las stopwords para el conteo
                if palabra[0] not in stopwords:
                    self.masRepetidasNOODIO.append(palabra)
            else:
                self.masRepetidasNOODIO.append(palabra)
        print(nComunes,"Palabras mas frecuentes en ODIO",self.masRepetidasODIO)
        print(nComunes,"Palabras mas frecuentes en NO ODIO",self.masRepetidasNOODIO)
        print()
        self.palabrasRepetidasODIONODIO=[] # Almacenamos las palabras mas repetidas en una lista común
        for elemento in self.masRepetidasODIO:
            self.palabrasRepetidasODIONODIO.append(elemento[0])
        for elemento in self.masRepetidasNOODIO:
            self.palabrasRepetidasODIONODIO.append(elemento[0])

    def procesamiento4(self, tenerEnCuentaStopWords):
        """
            - Descripción del procesamiento:
                - Elimina espacios vacíos irrelevantes del formateo .docx.
                - No se tienen en cuenta los carácteres no alfabéticos:
                    - Cabecera con el nombre del documento.
                    - Etiquetas de anonimizado.
                - Elimina los signos de puntuación.
                - Elimina las siguientes palabras de cada tipo de denuncia (Sin tener en cuenta stop words):
                    - Palabras comunes de cada clase, es decir, palabras presentes en todos los documentos de una misma clase.
                    - Palabras más repetidas de cada clase.
                    - Palabras exclusivas de cada clase, es decir, palabras que aparecen en todos los documentos de una misma clase y no aparecen en el otro tipo de denuncia.
            - Parámetros:
                - tenerEnCuentaStopWords: Booleano que indica si se tienen en cuenta las stops words en el cálculo de la bolsa de palabras de borrado
        """
        self.procesamiento2()
        self.procesamiento4CalcularPalabras(nComunes=500, tenerEnCuentaStopWords=tenerEnCuentaStopWords)
        self.procesamiento4AplicarPalabras(self.delitosnodio)
        self.procesamiento4AplicarPalabras(self.delitosodio)
    def procesamiento4AplicarPalabras(self, listaTextos): #Quitando puntos y caracteres no alfabéticos (nombre del primer parráfo y etiquetas de anonimizado)
        """
            Función para quitar la bolsa de palabras anteriormente calculada
            - Parámetros:
                - listaTextos: Lista de denuncias sobre las que aplicar el procesamiento
        """
        import string
        listaTextosAuxiliar=listaTextos
        # Se realiza los mismos primeros pasos que en el procesamiento 2
        i = 0
        for texto in listaTextos:
            lista = []
            for letra in texto:
                if letra not in string.punctuation: # Nos quedamos con las letras que no son signos de puntuación
                    lista.append(letra)
            texto = ''.join(lista) # Guardamos las letras en una cadena de texto
            palabras = texto.split() # Extraemos las palabras con un split
            # print(texto.split())
            listaAlfa = []
            # Nos quedamos con las palabras que no pertenecen a el cojunto de palabras exclusivas de cada clase, a las palabras más repetidas en ambas denuncias, ni a las palabras que aparecen en todos los archivos de cada tipo
            for palabra in palabras:
                        if palabra.lower() not in self.exclusivasNOODIO and palabra not in self.exclusivasODIO:
                            if palabra.lower() not in self.palabrasRepetidasODIONODIO:
                                if palabra.lower() not in self.palabrasComunesODIO and palabra.lower() not in self.palabrasComunesNOODIO:
                                    listaAlfa.append(palabra.lower())
            textoAlfa = ' '.join(listaAlfa) # Pasamos la lista de palabras a una cadena desparada por espacios

            listaTextosAuxiliar[i]=textoAlfa # Guardamos por referencia
            i = i + 1

        return listaTextosAuxiliar
    def procesamiento2PorParámetro(self, listaTextos): #Quitando puntos y caracteres no alfabéticos (nombre del primer parráfo y etiquetas de anonimizado)
        """
            Al tratarse el procesamiento elegido para el entrenamiento y prueba se
            necesitaba esta función para tratar textos recibidos por parámetro desde el controlador.
            Función para aplicar el procesamiento 2 a una sola lista.
            Parámetros:
                - listaTextos: Lista con el archivo a procesar en la primera posición
        """
        import string
        listaAuxiliar=listaTextos
        # Mismos pasos que en el procesamiento 2
        i = 0
        for texto in listaTextos:
            lista = []
            for letra in texto:
                if letra not in string.punctuation: # Quitamos los signos de puntuación
                    lista.append(letra)
            texto = ''.join(lista)
            palabras = texto.split() # Extraemos palabras con un split()
            # print(texto.split())
            listaAlfa = []
            for palabra in palabras:
                # print(palabra.isalpha(),palabra)
                if not palabra.isupper(): # Quitamos palabras que están completamente en mayúscula
                    if palabra.isalpha() or palabra.isnumeric(): # Nos quedamos con palabras alfabéticas y numéricas
                        listaAlfa.append(palabra.lower())
            textoAlfa = ' '.join(listaAlfa)

            listaAuxiliar[i]=textoAlfa # Guardamos
            i = i + 1

        return listaAuxiliar


@property
def getArchivosODIO(self):
    """
        Getter del la lista de denuncias de Odio.
        Devuelve:
                - Lista con las denuncias de Odio.
    """
    return self.delitosodio


@property
def getArchivosNOODIO(self):
    """
        Getter del la lista de denuncias de No Odio.
        Devuelve:
                - Lista con las denuncias de No Odio.
    """
    return self.delitosnodio
