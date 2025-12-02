"""
SepFPL Time Adaptive 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class SepFPLTimeAdaptive(BaseFactorizationTrainer):
    """SepFPL Time Adaptive 训练器：明确启用时间适应"""
    
    def get_factorization_name(self):
        return 'sepfpl_time_adaptive'

