import yaml
from pathlib import Path
from pipeline.config import Config

# Load default config
cfg = Config.from_yaml("default_config.yaml")

# Load best parameters from Optuna
with open("optuna_results/best_params.yaml", "r") as f:
    best_params = yaml.safe_load(f)

print("Applying Optuna best params to production config:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# Apply to model
if "fpcbam_reduction_ratio" in best_params:
    cfg.model.fpcbam_reduction_ratio = int(best_params["fpcbam_reduction_ratio"])

# Apply to loss
if "focal_alpha" in best_params:
    cfg.loss.focal_alpha = float(best_params["focal_alpha"])
if "focal_gamma" in best_params:
    cfg.loss.focal_gamma = float(best_params["focal_gamma"])
if "lambda_giou" in best_params:
    cfg.loss.lambda_giou = float(best_params["lambda_giou"])
if "lambda_shape" in best_params:
    cfg.loss.lambda_shape = float(best_params["lambda_shape"])

# Apply to matcher
if "center_cost_weight" in best_params:
    cfg.matcher.center_cost_weight = float(best_params["center_cost_weight"])
if "size_cost_weight" in best_params:
    cfg.matcher.size_cost_weight = float(best_params["size_cost_weight"])
if "giou_cost_weight" in best_params:
    cfg.matcher.giou_cost_weight = float(best_params["giou_cost_weight"])

# Apply to hardware
if "batch_size" in best_params:
    bs = int(best_params["batch_size"])
    cfg.hardware.micro_batch_size = bs
    cfg.hardware.gradient_accumulation_steps = max(1, cfg.hardware.target_effective_batch_size // bs)

# Apply to train
if "optimizer" in best_params:
    cfg.train.optimizer = str(best_params["optimizer"])
if "lr" in best_params:
    cfg.train.lr = float(best_params["lr"])
if "weight_decay" in best_params:
    cfg.train.weight_decay = float(best_params["weight_decay"])
if "sgd_momentum" in best_params:
    cfg.train.sgd_momentum = float(best_params["sgd_momentum"])
if "scheduler" in best_params:
    cfg.train.scheduler = str(best_params["scheduler"])
if "warmup_steps" in best_params:
    cfg.train.warmup_steps = int(best_params["warmup_steps"])
if "backbone_unfreeze_epoch" in best_params:
    cfg.train.backbone_unfreeze_epoch = int(best_params["backbone_unfreeze_epoch"])

cfg.train.num_epochs = 30
cfg.train.checkpoint_dir = "checkpoints"
cfg.hardware.log_file = "production_training.log"

cfg.to_yaml("production_config.yaml")
print("\nSaved production_config.yaml successfully!")
