import os

from flask import Flask, render_template, request, redirect, url_for
from transformers import pipeline, AutoTokenizer

from Pagina import clasificador


app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
@app.route("/index.html", methods=['GET', 'POST'])
def escribirTexto():
    """
        Función para escribir texto en la web.
        Devuelve:
            - El render de la vista index.html
    """
    os.chdir("ModelosDefinitivos")
    listaModelos = os.listdir()
    os.chdir('..')

    datos = { # Unidad de transporte de intercambio de datos entre el controlador y la vista.
        'titulo': 'Clasificador',
        'etiqueta': '',
        'score': '',
        'modelo': listaModelos[0],
        'error': '',
        'listaModelos': listaModelos
    }

    if request.method == "POST": # Recibimos el POST.
        print("POST recibido, contenido:", request.form.get('texto'))
        texto = request.form.get('texto') # Extraemos texto.
        modelo = request.form.get('modelo') # Estraemos modelo.

        datos['listaModelos']=clasificador.setUltimoModeloUsado(listaModelos,modelo)
        datos['modelo'] = modelo
        print("modelo",modelo)
        if texto!="": # Clasificamos el texto.
            etiqueta,score=clasificador.clasificar(texto, modelo)
            datos['etiqueta']=etiqueta
            datos['score']=score
        else:
            datos['error'] = "Debe introducir un texto"
    return render_template('index.html', data=datos)


@app.route("/subirArchivo.html", methods=['GET', 'POST'])
def subirArchivo():
    """
        Función para subir un archivo docx en la web.
        Devuelve:
            - El render de la vista subirArchivo.html
    """
    os.chdir("ModelosDefinitivos")
    listaModelos = os.listdir()
    os.chdir('..')
    datos = { # Unidad de transporte de intercambio de datos entre el controlador y la vista.
        'titulo': 'Clasificador',
        'etiqueta': '',
        'score': '',
        'modelo': listaModelos[0],
        'error': '',
        'listaModelos': listaModelos
    }
    if request.method == "POST": # Recibimos el POST.
        archivo = request.files['file']
        modelo = request.form.get('modelo')
        datos['listaModelos'] = clasificador.setUltimoModeloUsado(listaModelos,modelo)
        datos['modelo'] = modelo
        print(modelo)
        if archivo.filename != "": # Comprobamos si hay archivo subido.
            archivo.save(archivo.filename)
            import docx2txt
            import glob  # Lectura y procesado de los archivos.
            formatoValido = False
            for filename in glob.glob('*.docx'):
                with open(os.path.join(os.getcwd(), filename), 'r') as f:
                    texto = docx2txt.process(filename) # Extraemos el texto del documento.
                    if (texto != ""): # Comprobamos si el archivo no está vacío.
                        print(filename, " Leído")
                        # print(texto)
                        etiqueta, score = clasificador.clasificar(texto, modelo)
                        datos['etiqueta'] = etiqueta
                        datos['score'] = score
                        print(etiqueta, score)
                        print(os.getcwd())
                formatoValido = True
                os.remove(archivo.filename)
            if formatoValido == False:
                datos['error'] = "Solo se acepta formato .docx"
        else:
            datos['error'] = "Debe introducir un archivo .docx"

    return render_template('subirArchivo.html', data=datos)



def query_string():
    print(request)
    print(request.args)
    print(request.args.get('param'))
    return "Ok"


def error(error):
    os.chdir("ModelosDefinitivos")
    listaModelos = os.listdir()
    os.chdir('..')
    envio = {
        'titulo': 'Clasificador',
        'modelos': '',
        'listaModelos': listaModelos
    }
    envio['modelos'] = ""
    return render_template('paginaError.html', data=envio), 404
    # return redirect(url_for('index'))


if __name__ == '__main__':
    app.add_url_rule('/consulta', view_func=query_string)  # Enlazamos la funcion a la url forma, distinta a la anterior.
    app.register_error_handler(404, error)  # Manejador del error.

    app.run(debug=True, port=5000)
