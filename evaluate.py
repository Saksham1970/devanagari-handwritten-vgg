from utils.hparams import setup_hparams
from utils.setup_net import setup_net
from train import run


hparams = {
    "net": "VGG_8",
    "name": "VGG_8_Devanagari_SDG_RLRP_min",
    "batch_size": 64,
    "n_epochs": 150,
    "database_path": "./Devanagari",
    "num_classes": 46,
}

hparams = setup_hparams(**hparams)
logger, net = setup_net(hparams)


run(net, logger, hparams)
