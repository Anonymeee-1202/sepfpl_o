"""
SepFPL HCSE 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class SepFPLHCSE(BaseFactorizationTrainer):
    """SepFPL HCSE 训练器：明确启用 HCSE"""
    
    def get_factorization_name(self):
        return 'sepfpl_hcse'

