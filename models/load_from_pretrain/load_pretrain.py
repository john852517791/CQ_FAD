import sys
sys.path.append('./')
import torch
def load(
    file="a_train_log/pretrain/version_7/checkpoints/best_model-epoch=58-dev_eer=0.0000.ckpt"
    ):
    print(f"load{file}")
    return torch.load(file)


# "a_train_log/pretrain/version_2/checkpoints/best_model-epoch=44-dev_eer=0.0000.ckpt"
# "a_train_log/pretrain/version_3/checkpoints/best_model-epoch=29-dev_eer=0.0000.ckpt"
# "a_train_log/pretrain/version_8/checkpoints/best_model-epoch=58-dev_eer=0.0000.ckpt"
# "a_train_log/pretrain/version_7/checkpoints/best_model-epoch=58-dev_eer=0.0000.ckpt"


# z_train/pretrain/16/version_1/checkpoints/best_model-epoch=04-batch_loss=0.270785.ckpt
# z_train/pretrain/8/version_0/checkpoints/best_model-epoch=09-batch_loss=0.050563.ckpt
# z_train/pretrain/2/version_0/checkpoints/best_model-epoch=09-batch_loss=0.109829.ckpt
