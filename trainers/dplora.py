"""
DPLoRA 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class DPLoRA(BaseFactorizationTrainer):
    """DPLoRA 训练器：global_ctx + local_u_ctx @ local_v_ctx (无 residual)"""
    
    def get_factorization_name(self):
        return 'dplora'

