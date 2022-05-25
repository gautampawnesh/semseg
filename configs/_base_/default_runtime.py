
# validation setup
evaluation = dict(interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best='mIoU')
workflow = [("train", 1)]
# launcher
launcher = "slurm"
# distributed params
dist_params = dict(backend="nccl")

# Logging configuration
log_config = dict(
    interval=25,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook")
    ],
)
cudnn_benchmark = True
