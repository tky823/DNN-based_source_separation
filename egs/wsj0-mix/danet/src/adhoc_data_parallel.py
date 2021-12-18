import threading
from itertools import chain

import torch
import torch.nn as nn
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper
from torch.nn.parallel.parallel_apply import get_a_var

class AdhocDataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

    def extract_latent(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DataParallel.extract_latent"):
            if not self.device_ids:
                return self.module.extract_latent(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                if t.device != self.src_device_obj:
                    raise RuntimeError("module must have its parameters and buffers on device {} (device_ids[0]) but found one of them on device: {}".format(self.src_device_obj, t.device))

            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            # for forward function without any inputs, empty list and dict will be created
            # so the module can be executed on one device which is the first one in device_ids
            if not inputs and not kwargs:
                inputs = ((),)
                kwargs = ({},)

            if len(self.device_ids) == 1:
                return self.module.extract_latent(*inputs[0], **kwargs[0])

            replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
            outputs = self.parallel_apply_extract_latent(replicas, inputs, kwargs)

            return self.gather(outputs, self.output_device)

    def parallel_apply_extract_latent(self, replicas, inputs, kwargs):
        return parallel_apply_extract_latent(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

"""
    Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/parallel/parallel_apply.py
"""
def parallel_apply_extract_latent(modules, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.
    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices
    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)

    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    devices = [_get_device_index(x, True) for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(i, module, input, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = module.extract_latent(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []

    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    
    return outputs