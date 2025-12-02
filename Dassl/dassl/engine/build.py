from Dassl.dassl.utils import Registry, check_availability
from utils.logger import require_global_logger

# 导入所有训练器类
from trainers.promptfl import PromptFL
from trainers.fedotp import FedOTP
from trainers.fedpgp import FedPGP
from trainers.dplora import DPLoRA
from trainers.dpfpl import DPFPL
from trainers.sepfpl import SepFPL
from trainers.sepfpl_time_adaptive import SepFPLTimeAdaptive
from trainers.sepfpl_hcse import SepFPLHCSE

TRAINER_REGISTRY = Registry("TRAINER")

# 注册所有训练器
TRAINER_REGISTRY.register(PromptFL)
TRAINER_REGISTRY.register(FedOTP)
TRAINER_REGISTRY.register(FedPGP)
TRAINER_REGISTRY.register(DPLoRA)
TRAINER_REGISTRY.register(DPFPL)
TRAINER_REGISTRY.register(SepFPL)
TRAINER_REGISTRY.register(SepFPLTimeAdaptive)
TRAINER_REGISTRY.register(SepFPLHCSE)

# 为了向后兼容，手动注册 'SEPFPL' 作为 'SepFPL' 的别名
# 因为 federated_main.py 和 mia.py 中使用了 'SEPFPL' 字符串
TRAINER_REGISTRY._obj_map['SEPFPL'] = SepFPL

def build_trainer(cfg):
    """
    构建训练器
    
    如果 cfg.TRAINER.NAME 存在，使用它；否则根据 cfg.FACTORIZATION 自动选择
    """
    # 如果指定了 TRAINER.NAME，使用它
    if hasattr(cfg, 'TRAINER') and hasattr(cfg.TRAINER, 'NAME') and cfg.TRAINER.NAME:
        trainer_name = cfg.TRAINER.NAME
    else:
        # 否则根据 FACTORIZATION 自动选择
        factorization_to_trainer = {
            'promptfl': 'PromptFL',
            'fedotp': 'FedOTP',
            'fedpgp': 'FedPGP',
            'dplora': 'DPLoRA',
            'dpfpl': 'DPFPL',
            'sepfpl': 'SepFPL',
            'sepfpl_time_adaptive': 'SepFPLTimeAdaptive',
            'sepfpl_hcse': 'SepFPLHCSE',
        }
        
        if hasattr(cfg, 'FACTORIZATION') and cfg.FACTORIZATION in factorization_to_trainer:
            trainer_name = factorization_to_trainer[cfg.FACTORIZATION]
            # 更新配置以便后续使用
            if not hasattr(cfg.TRAINER, 'NAME') or not cfg.TRAINER.NAME:
                cfg.TRAINER.NAME = trainer_name
        else:
            # 默认使用 SepFPL
            trainer_name = 'SepFPL'
            if not hasattr(cfg.TRAINER, 'NAME') or not cfg.TRAINER.NAME:
                cfg.TRAINER.NAME = trainer_name
    
    avai_trainers = TRAINER_REGISTRY.registered_names()
    check_availability(trainer_name, avai_trainers)
    if cfg.VERBOSE:
        logger = require_global_logger()
        logger.info("Loading trainer: {} (factorization: {})".format(
            trainer_name, 
            getattr(cfg, 'FACTORIZATION', 'N/A')
        ))
    return TRAINER_REGISTRY.get(trainer_name)(cfg)
