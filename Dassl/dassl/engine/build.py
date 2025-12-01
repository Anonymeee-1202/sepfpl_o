from Dassl.dassl.utils import Registry, check_availability
from trainers.DP_FPL import SEPFPL
from utils.logger import require_global_logger

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(SEPFPL)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        logger = require_global_logger()
        logger.info("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
