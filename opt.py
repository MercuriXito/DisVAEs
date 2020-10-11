import argparse
import torch

import warnings

from models.annealvae import AnnealVAE
from models.vanilla_vae import VanillaVAE
from models.dipvae import DipVAE
from models.betavae import BetaVAE

from utils import read_config_from_yaml


_integrity_list = [
    # name, type, default, required, help

    # # model related
    # ["model_name", str, None, True, "name of model"],
    # ["latent_dim", int, 20, False, "latent dim"],
    # ["hidden_channels", list, [64, 128, 256], False, ""],
    # ["in_channels", int, 1, False, "input channels"],

    # data related
    ["data_name", str, "mnist", False, ""],
    ["data_path", str, "save", False, ""],
    ["batch_size", int, 16, False, "batch size"],
    ["shuffle", bool, True, False, ""],
    ["grid_nrow", int, 16, False, ""],
    ["num_workers", int, 4, False, ""],
    ["input_size", int, 64, False, ""],

    # training related
    ["epochs", int, 30, False, ""],
    ["lr", float, 1e-4, False, ""],
    ["use_cuda", bool, True, False, "using cuda"],
    ["optimizer", str, "adam", False, ""],
    ["betas", list, [0.5, 0.999], False, ""],
    ["momentum", float, 0.0, False, ""],
    ["lr_decay_weight", float, 0.5, False, ""],
    ["decay_step", int, 5, False, ""],
    ["num_epoch_save", int, 1, False, ""],

    # resume related
    ["resume", bool, False, False, ""],
    ["resume_folder", str, "", False, ""],
]

def get_options():
    parser = argparse.ArgumentParser()
    for name, _type, _default, _required, _help in _integrity_list:
        parser.add_argument("--{}".format(name.replace("_","-")), 
            type=_type,
            default=_default,
            required=_required,
            help=_help
        )
    return parser.parse_args()

def _get_config():
    """ get yaml config file """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="specify the yaml config file")
    return parser.parse_args()

def _check_integrity(config: dict):
    """ check the type and assign default values """
    keys = list(config.keys())
    for name, _type, _default, _required, _ in _integrity_list:
        if name in keys:
            assert isinstance(config[name], _type), "config key \"{}\" not in type: {}".format(name, _type)
        elif _required:
            raise KeyError("Required keyword: {}".format(name))
        else:
            warnings.warn("Key {} does not exist, setting to default value:{}".format(name, _default))
            config[name] = _default
    return config

def get_config():
    """ read yaml config file and convert to option"""
    config_file = _get_config().config
    config = read_config_from_yaml(config_file)
    config = _check_integrity(config)
    opt = _MetaOptions.dict2opts(config)
    return opt

def get_model(opt):
    model_name = opt.model_params.model_name
    model_params = opt.model_params._get_kwargs()
    model_params["input_size"] = opt.input_size

    if model_name == "vanilla_vae":
        vae = VanillaVAE(**model_params)
    elif model_name == "factorvae":
        vae = FactorVAE(**model_params)
    elif model_name == "betavae":
        vae = BetaVAE(**model_params)
    elif model_name == "annealvae":
        vae = AnnealVAE(**model_params)
    elif model_name == "dipvae":
        vae = DipVAE(**model_params)
    else:
        raise NotImplementedError("Not implemented model: {}".format(model_name))

    return vae

class _MetaOptions:
    """ Options-like object: providing functions to convert between option and dict. """
    # def __str__(self, level=0):
    #     """ Beautified ouput like dict-object """
    #     _get_str = lambda val: val.__str__(level+1) if isinstance(val, _MetaOptions) else val.__str__()
    #     spaces = " " *(level+1)*4
    #     less_spaces = " "*level*4
    #     return "{\n%s\n%s}" %( 
    #             ",\n".join(["{}{}:{}".format(spaces, key,_get_str(val)) for key, val in self.__dict__.items()]),
    #             less_spaces
    #     ) 

    def __str__(self):
        return "{}".format(self._get_kwargs())

    @staticmethod
    def kws2opts(**kws):
        """ Recursively convert all keyword input to option like object. Useful to convert 
        functions that only admit option input to those with kwargs input.
        """
        return _MetaOptions.dict2opts(kws)

    @staticmethod
    def dict2opts(d: dict):
        """ Recursively convert dict to option like object (_MetaOptions).
        """
        o = _MetaOptions()
        def _parse(obj, dt: dict):
            for key, val in dt.items():
                if not isinstance(key, str):
                    raise AttributeError("Not allowed key in dict with type:{}".format(type(key)))
                if isinstance(val, dict):
                    t = _MetaOptions()
                    setattr(obj, key, t)
                    _parse(t, val)
                else:
                    setattr(obj, key, val)
            return obj
        return _parse(o, d)

    def _get_kwargs(self):
        """ opt to dict """
        _sub_get_kwargs = lambda x: x._get_kwargs() if isinstance(x, _MetaOptions) else x
        return dict([ (key, _sub_get_kwargs(val)) for key, val in self.__dict__.items() ])

if __name__ == '__main__':

    from data import get_loader
    from utils import TensorImageUtils
    import matplotlib.pyplot as plt

    opt = get_config()
    print(opt)
    vae = get_model(opt)
    loader = get_loader(opt)

    preprocess_func = lambda x: x
    utiler = TensorImageUtils(preprocess_func=preprocess_func, normalize=False)

    for i, batch in enumerate(loader):
        batch = batch[0]
        print(batch.size())
        print(batch.sum())
        utiler.plot_show(batch, nrow=opt.grid_nrow)
        plt.show()
        break
    pass


