import sys
from parsing_by_maxseminfo import parser
sys.modules['parser'] = parser
from .utils.myargparse import get_argsndevice
from parser.lightning_wrapper.LitNPCFG import (
    LitXNPCFGFCReward,
)
from parser.helper.pas_grammar_data_helper import (
    DataModuleForPASCtrlPCFGReward,
)
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint


args, device = get_argsndevice()

print("continue training from", args.continue_from)



# %%

derivative = args.model.model_name.split("-")[1]
if derivative == "FixedCostReward":
    dst = DataModuleForPASCtrlPCFGReward(
        hparams=args,
        langstr=args.langstr,
        use_cache=True,
        max_size=10000,
        merge_pas_data=False,
        pas_subsample=args.preprocessing_pas_subsample_count,
        flag_use_pos_unks=(
            args.experimental.flag_use_pos_unks
            if hasattr(args.experimental, "flag_use_pos_unks")
            else False
        ),
    )
else:
    raise NotImplementedError(f"Derivative must be FixedCostReward, current derivative is {derivative}")

word_vocab = dst.word_vocab

basemodel = args.model.model_name.split("-")[0]

print(f"launching {args.model.model_name.split('-')}")


if basemodel in ["SNPCFG", "TNPCFG", "NPCFG", "CPCFG", "SCPCFG", "SNPCFGA2C", "NPCFGA2C", "CPCFGA2C"]:
    derivative = args.model.model_name.split("-")[1]
    print(f"launching {basemodel} {derivative}")
    if derivative == "FixedCostReward":
        model = LitXNPCFGFCReward(
            basemodel,
            args.model,
            word_vocab.vocab_size,
            args.experimental,
            args.optimizer,
            args.langstr,
        )
    else:
        raise NotImplementedError(f"{derivative} is not allowed")
else:
    raise NotImplementedError(f"{args.model.model_name} is not allowed")




tensorboard_logger = TensorBoardLogger(
    "lightning_logs", name=f"{args.wandb_project}/{args.remark}"
)
args_to_log = {
    "train": args.train,
    "model": args.model,
    "optim": args.optimizer,
    "experimental": args.experimental,
}
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(
    project=args.wandb_project,
    name=f"{args.remark}",
    log_model=False,
    entity="junjie-playground",
    config=args_to_log,
    tags=args.wandb_tags,
)
wandb_logger.watch(model, log_graph=False, log_freq=100)


# Setup early stopping
early_stop_callback = EarlyStopping(
    monitor="val/sentence_f1",  # Metric to monitor
    min_delta=0.002,  # Minimum change to qualify as an improvement
    patience=args.train.patience,  # Number of epochs with no improvement after which training will be stopped
    verbose=True,
    mode="max",  # Minimize the monitored metric (use 'max' for metrics like accuracy)
)

train_dl, _ = dst.train_dataloader(
        args.langstr,
        max_len=args.max_length,
        min_len=3,
        device=device,
        pas_subsample_count=args.experimental.pas_subsample_count,
        flag_curriculum_learning=(
            args.experimental.flag_curriculum_learning
            if hasattr(args.experimental, "flag_curriculum_learning")
            else False
        ),
        add_sentence_level_span=(
            args.experimental.add_sentence_level_span
            if hasattr(args.experimental, "add_sentence_level_span")
            else False
        ),
        min_span_reward=args.experimental.min_span_reward,  # min span reward must be specified
        mode_reward=(
            args.experimental.mode_reward
            if hasattr(args.experimental, "mode_reward")
            else "log_tfidf"
        ),
        supervised_mode=(
            args.experimental.supervised_mode
            if hasattr(args.experimental, "supervised_mode")
            else False
        ),
    )

val_dl, _ = dst.dev_full_dataloader(
    args.langstr,
    max_len=100000,
    min_len=2,
    device=device,
    min_span_reward=args.experimental.min_span_reward,
    mode_reward=(
        args.experimental.mode_reward
        if hasattr(args.experimental, "mode_reward")
        else "log_tfidf"
    ),
)

test_dl, _ = dst.test_dataloader(
    args.langstr,
    max_len=1000000,
    min_len=2,
    device=device,
)


best_sf1_checkpoint_callback = ModelCheckpoint(
    save_top_k=4,
    monitor="val/sentence_f1",
    mode="max",
    dirpath=args.ckpt_dir,
    filename="ckpt-sf1_{val/sentence_f1:.2f}",
)
saveall_checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,
    dirpath=args.ckpt_dir,
    filename="ckpt-step_{step}",
)


from parser.lightning_wrapper.scheduler import WarmupScheduler

rl_coeff_scheduler = WarmupScheduler(
    warmup_steps=(
        args.experimental.rl_warmup_steps
        if hasattr(args.experimental, "rl_warmup_steps")
        else 10000
    ),
    coeff_name="rl_coeff",
    initial_coeff=(
        args.experimental.rl_initial_coeff
        if hasattr(args.experimental, "rl_initial_coeff")
        else 0.0
    ),
    start_step=(
        args.experimental.rl_start_step
        if hasattr(args.experimental, "rl_start_step")
        else 20000
    ),
    target_coeff=(
        args.experimental.rl_target_coeff
        if hasattr(args.experimental, "rl_target_coeff")
        else 0.3
    ),
)

maxent_scheduler = WarmupScheduler(
    warmup_steps=(
        args.experimental.maxent_warmup_steps
        if hasattr(args.experimental, "maxent_warmup_steps")
        else 1
    ),
    coeff_name="maxent_coeff",
    initial_coeff=(
        args.experimental.maxent_initial_coeff
        if hasattr(args.experimental, "maxent_initial_coeff")
        else 0.5
    ),
    start_step=(
        args.experimental.maxent_start_step
        if hasattr(args.experimental, "maxent_start_step")
        else 0.0
    ),
    target_coeff=(
        args.experimental.maxent_target_coeff
        if hasattr(args.experimental, "maxent_target_coeff")
        else 0.5
    ),
)

max_steps = 100000
min_steps = 30000
val_check_interval = args.val_check_interval

trainer = L.Trainer(
    max_steps=max_steps,
    min_steps=min_steps,
    min_epochs=0,
    val_check_interval=val_check_interval,
    check_val_every_n_epoch=None,
    gradient_clip_val=args.train.clip,
    gradient_clip_algorithm="norm",
    callbacks=[
        early_stop_callback,
        TQDMProgressBar(refresh_rate=50),
        best_sf1_checkpoint_callback if not args.analysis_mode and not args.corr_mode else saveall_checkpoint_callback,
        rl_coeff_scheduler,
        maxent_scheduler,
    ],
    logger=[wandb_logger, tensorboard_logger],  # if not args.debug else None,
    inference_mode=False,
    log_every_n_steps=10,
    accelerator="gpu",
    devices=args.ngpu,          # Number of GPUs to use
    strategy="ddp"      # Use Distributed Data Parallel
    
)
# wandb_logger.watch(model, log_graph=False, log_freq=100)
trainer.fit(
    model,
    train_dataloaders=train_dl,
    val_dataloaders=val_dl,
    ckpt_path=args.continue_from,
)

print(
    "Training ends. The best model on Spacy mode: \n",
    trainer.test(model, dataloaders=test_dl),
    file=sys.stderr,
)
