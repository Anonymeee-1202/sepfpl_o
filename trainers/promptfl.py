"""
PromptFL 训练器
"""
from trainers.base_trainer import BaseFactorizationTrainer


class PromptFL(BaseFactorizationTrainer):
    """PromptFL 训练器：仅使用 global_ctx"""
    
    def get_factorization_name(self):
        return 'promptfl'

