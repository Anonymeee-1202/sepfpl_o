"""
DPFPL 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class DPFPL(BaseFactorizationTrainer):
    """DPFPL 训练器：global_ctx + local_u_ctx @ local_v_ctx + residual"""
    
    def get_factorization_name(self):
        return 'dpfpl'

