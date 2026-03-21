# Detalles de implementación y funcionamiento

## Bases
### Pytorch

Los modelos son instancias de la clase ```nn.Module```. PyTorch organiza las redes como un árbol de submódulos. Cada modelo es un objeto que contiene un   ```OrderedDict``` de capas.

## Definiendo datos sobre 

* ```blocklize/block_meta.py``` : Se define, de forma manual, cuales son los bloques mínimos internos de las redes.

    ```python
    MODEL_ZOO = ['resnet18', 'mobilenetv3_small_050', 'shufflenet_v2_x0_5', 'squeezenet1_1', 'efficientnet_b0']

    MODEL_BLOCKS = {
        'resnet18': ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1'],
        'mobilenetv3_small_050': ['blocks.0.0', 'blocks.1.0', 'blocks.1.1', 'blocks.2.0', 'blocks.2.1', 'blocks.2.2', 'blocks.3.0', 'blocks.3.1', 'blocks.4.0', 'blocks.4.1', 'blocks.4.2', 'blocks.5.0'],
        # ... Más modelos
    ```

    * **MODEL_ZOO** :  Cada elemento representa el nombre de un modelo conocido, con su respectiva arquitectura. Los modelos enlistados aquí son objectos extraídos desde **Torchvision** o **TIMM** 

    <!-- Como es que son importados en código?-->

    * Se procesan/inspeccionan las arquitecturas PyTorch y se identifican los nombres de las capas secuenciales que forman los bloques lógicos.
    
    * **MODEL_BLOCKS** : Los calores en esta lista correponden a los nombres de los sub-módulos 

    



<!--  

De donde se saca Model_Blocks ?

De donde saca los modelos model_zoo?



-->