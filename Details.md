# Detalles de implementación y funcionamiento

## Bases
### Pytorch

Los modelos son instancias de la clase ```nn.Module```. PyTorch organiza las redes como un árbol de submódulos. Cada modelo es un objeto que contiene un   ```OrderedDict``` de capas.

## Definiendo datos sobre las redes

Todo comienza en el archivo ```blocklize/block_meta.py```, donde se establece el "mapa" de lo que el sistema puede procesar.
<!-- 
FALTA: 
    * Como se representa una arquitectura de red en PyTorch?
    * Rol del archivo Model_In_OUt.
-->

* ```blocklize/block_meta.py``` : Se define, de forma manual, cuales son los bloques mínimos internos de las redes.

    * **MODEL_ZOO** :  Lista que contiene los identificadores de las arquitecturas compatibles.  Los modelos enlistados aquí son objectos extraídos desde **Torchvision** o **TIMM** y poseen su propia arquitectura. 

    * Se procesan/inspeccionan las arquitecturas PyTorch y se identifican los nombres de las capas secuenciales que forman los bloques lógicos mínimos utilizables. Esto se hace para no particionar las redes en porciones más pequeñas que rompan con el flujo de datos de la arquitectura en puntos críticos *(ej. Skip connections)*

    * **MODEL_BLOCKS** : Diccionario que mapea el nombre de cada modelo con sus puntos de corte posibles, es decir, que define los "puntos de sutura" para cada modelo.  Estos nombres sin identificadores internos de PyTorch/TiMM para las capas, donde se ha identificado donde termina un bloque l+ogico de procesamiento

    Los valores de las lisyas corresponden a los nombres de los submódulos en el árbol de cómputo ```nn.Module``` del modelo. 

## Carga dinámica de modelos
Antes de fragmentar un modelo, el sistema debe instanciarlo con sus conocimientos previos *(pesos pre-entrenados)*. Esto ocurre principalmente en la clase ```SuperNetwork``` dentro de ```simple_poc/supernet.py```.

* Proceso (```_extract_submodule```):

    1. El código identifica si el modelo pertenece a **torchvision** *(como resnet o densenet)* o si requiere la librería **timm** *(como efficientnet)*.

    2. Llama a la función correspondiente (ej. ```models.resnet18(weights='DEFAULT')``` o timm.```create_model(name, pretrained=True))```.

* **Entrada**: El nombre del modelo (string).

* **Salida**: Un objeto ````nn.Module```` que contiene la red completa y sus pesos entrenados en ImageNet.
    
## Partiendo las redes

Se ejecuta en la función ````create_sub_network```` del archivo ````simlarity/feature_extraction.py````. Aquí, el modelo completo se *rebana* para extraer solo la sección deseada.

Los modelos de PyTorch no pueden ser partidos usando indexación *(ej. ```model[:5]```)* porque pueden tener topologías complejas, *(ej. Skip connections)*. Para solucionar esto, se usa **PyTorch FX**, una herramienta para transformar instancias de ```nn.Module```

* **Paso 1**: *Trazado Simbólico* (``symbolic_trace``): **PyTorch FX** analiza el código del modelo y genera un grafo (``torch.fx.Graph``) que representa cada operación matemática como un **Nodo**.

* **Paso 2:** *Identificación de Nodos*: La función recibe ``input_args`` (donde empieza el bloque) y ``return_nodes`` (donde termina). El código recorre el grafo completo para localizar estos puntos exactos por nombre.

* **Paso 3**: *Aislamiento del Sub-grafo:*

    1. Se crea un nuevo grafo vacío.

    2. El nodo de inicio se convierte en un ``placeholder``, lo que significa que ahora el bloque puede recibir datos externos directamente en ese punto.

    3. Se copian todas las operaciones intermedias (convoluciones, activaciones, conexiones residuales) que conectan el inicio con el fin.

    4. El nodo final se marca como la salida única del bloque (``output``).

**Entradas**:

* **``model``**: El objeto del modelo completo.

* **``input_args``**: Lista con el nombre del nodo donde comienza la extracción.

* **``return_nodes``**: Lista con el nombre del nodo donde termina la extracción.

**Salida**: Un ``GraphModule``. Este es un componente de PyTorch que se comporta como una red neuronal independiente, conteniendo solo la lógica y los pesos de esa sección específica del modelo original.

