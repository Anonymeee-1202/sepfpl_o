from Dassl.dassl.utils import Registry, check_availability
from trainers.DP_FPL import DP_FPL
from utils.logger import get_global_logger, get_logger

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.register(DP_FPL)

def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        logger = get_global_logger() or get_logger('dp-fpl', log_dir='logs', log_to_file=False, log_to_console=True)
        logger.info("Loading trainer: {}".format(cfg.TRAINER.NAME))
    return TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
