import smdebug.pytorch as smd
import os
out_dir = "/home/ubuntu/resnet-sagemaker/pytorch/debugger_logs/"


# delete out_dir if it already exists
if os.path.exists(out_dir):
    os.system("rm -rf " + out_dir)
save_config = smd.SaveConfig()
hook = smd.Hook(
    out_dir=out_dir,
    export_tensorboard=True,
    save_config=save_config
)