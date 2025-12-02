"""
FedOTP 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class FedOTP(BaseFactorizationTrainer):
    """FedOTP 训练器：global_ctx + local_ctx"""
    
    def get_factorization_name(self):
        return 'fedotp'

