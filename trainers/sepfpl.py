"""
SepFPL 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class SepFPL(BaseFactorizationTrainer):
    """SepFPL 训练器：基于 DPFPL，支持时间适应和 HCSE"""
    
    def get_factorization_name(self):
        return 'sepfpl'

