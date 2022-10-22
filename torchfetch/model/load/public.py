from torchvision import models


def is_public_architecture(arch_name: str) -> bool:
    return is_torchvision_architecture(arch_name) # or others in the future

def is_torchvision_architecture(arch_name:str) -> bool:
    return hasattr(models, arch_name)

def instantiate_public_network(arch_name:str, **kwargs):
    network_class = getattr(models, arch_name)
    return network_class(**kwargs)
    
