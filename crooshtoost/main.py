"""
main.py exposes a public interface for setting up and managing a CTKernel object indirectly
since directly editing the global CTKernel is not recommended.
"""

from . import kernel

__KERNEL__ = kernel.CTKernel()

def set_model(model):
    global __KERNEL__
    __KERNEL__.set_model(model)

def run(
    initial_settings=None,
    data_downloader=None,
    data_processor=None,
    train_setup=None):
    global __KERNEL__
    
    __KERNEL__.run()