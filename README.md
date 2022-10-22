# torchfetch

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
![example workflow](https://github.com/jaeminSon/torchfetch/actions/workflows/test.yml/badge.svg)

### Install
``` 
pip install https://github.com/jaeminSon/torchfetch
(github ID, password 입력 필요)
```
### Usage
``` 
>>> import torchfetch

# get dataloader (classification)
>>> kwargs_dataloader = {"num_workers": 16, "pin_memory": True, "batch_size": 32, "shuffle": True}
>>> dataloader = torchfetch.get_dataloader(data="Test/objects/data/image_csv",   
                                           preprocess="Test/objects/preprocess/imagenet.json",  
                                           augment="Test/objects/augment/imagenet.json", 
                                           **kwargs_dataloader)  
>>> for batch in dataloader:
...     <process batch>
    
# get dataloader (detection)
>>> kwargs_dataloader = {"num_workers": 16, "pin_memory": True, "batch_size": 16, "shuffle": True, "collate_fn": lambda x: x}
>>> dataloader = torchfetch.get_dataloader(data="Test/objects/data/detection1",  
                                           preprocess="Test/objects/preprocess/detection1.json",  
                                           augment="Test/objects/augment/cocodetection.json", 
                                           **kwargs_dataloader)

# public network class instantiation
>>> network = torchfetch.instantiate_network("resnet50")
>>> network = torchfetch.instantiate_network("resnet34", **{"pretrained":True})

# load a network from model name (or recipe name for private architecture)
>>> network = torchfetch.get_pretrained_network("Test/objects/recipe/private_arch.json")

# network class instantiation
>>> network = torchfetch.instantiate_network("Test/objects/recipe/private_arch.json")

# get checkpoint
>>> checkpoint = torchfetch.get_checkpoint("Test/objects/recipe/private_arch.json")

# get network state dict
>>> model_state_dict = torchfetch.get_model_state_dict("Test/objects/recipe/private_arch.json")

# get optimizer state dict
>>> optimizer_state_dict = torchfetch.get_optimizer_state_dict("Test/objects/recipe/private_arch.json")

```

### File structure
```
# install graphviz and pydeps
linux: $ sudo apt install graphviz
mac: $ brew install graphviz

$ pip install pydeps

# draw dependency graph
$ pydeps torchfetch --reverse --only torchfetch --exclude-exact torchfetch

# No cycle found with the following command
$ pydeps torchfetch --reverse --only torchfetch --exclude-exact torchfetch --show-cycles

```
<img src="./torchfetch.svg" width="600">


### Custom data example (classification with image and annotation json file)
<img width="600" src="https://user-images.githubusercontent.com/8290383/186800847-6f2aa6bc-e342-43d0-8990-da3545a1365d.png">

## Custom data types
<img width="800" src="https://user-images.githubusercontent.com/8290383/186800883-bbaaf8bc-fa51-4849-9798-d5acc1d62dfb.png">  
<img width="800" src="https://user-images.githubusercontent.com/8290383/186800901-9336862c-a7e6-455f-9463-8baf8914dfad.png"> 
<img width="800" src="https://user-images.githubusercontent.com/8290383/186800923-25210f0d-602b-44bd-9251-00ad4c510a77.png">  
<img width="800" src="https://user-images.githubusercontent.com/8290383/186800949-f72b9498-f622-4803-8f5c-7c2eeeb9b5ec.png"> 
<img width="800" src="https://user-images.githubusercontent.com/8290383/186800966-54f908ee-6a2c-46cd-a462-a2f8dfff1ec8.png">
<img width="800" src="https://user-images.githubusercontent.com/8290383/186800984-8a8d8af9-8978-4de3-9e1b-ed1ff3eaa28d.png">
