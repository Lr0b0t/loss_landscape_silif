import os
import cifar10.model_loader
from sparch1.sparch.models.snns import SNN
import torch
import sys
# Add the directory containing the sparch module to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sparch1')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'sparch1/sparch')))
print("Python module search paths:")
for path in sys.path:
    print(path)
import sparch.models  


def load(args, dataset, model_name, model_file, data_parallel=False):
    batch_size = 64 
    nb_inputs = 700
    nb_hiddens = args.nb_hiddens #128
    nb_layers = args.nb_layers # 2
    nb_outputs = 20 if dataset == "SHD" else 35
    pdrop =  args.pdrop # 0.1

    input_shape = (batch_size, None, nb_inputs)
    layer_sizes = [nb_hiddens] * (nb_layers - 1) + [nb_outputs]

    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'SHD' or dataset == "SSC" or dataset == "SC":
        net = SNN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                neuron_type=model_name,
                dropout=pdrop,
                normalization=True,
                use_bias=False,
                bidirectional=False,
                use_readout_layer=True,
                extra_features={'rst_detach':False, 'time_offset':0, 'residual':False,
                                'superspike':False,'slayer':False,'xavier_init':False,
                                'clamp_alpha':False,'no_reset':False,'complex_reset':False,
                                'rst_detach':False, 'half_reset':True, 'dt_min':0.01 , 'dt_max':0.5,
                                'no_b':False
                                }
            ) 
        # Load the model
        net = torch.load(model_file)


 
    return net


 

 