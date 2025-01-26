import argparse

# from numpy import require
import torch
import yaml
from easydict import EasyDict as edict


def get_argsndevice(force_args=[]):
    parser = argparse.ArgumentParser(description="pcfgs")
    parser.add_argument(
        "--conf", "-c", default="config/npcfg_nt30_t60_curriculum0.yaml"
    )
    # parser.add_argument('--device', '-d', default='0')
    parser.add_argument("--use_tf32", action="store_true")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--ngpu", type=int, default=1)
    # parser.add_argument("--use_simlen_pasdata", action="store_true")
    # parser.add_argument("--merge_pas_data", action="store_true")
    parser.add_argument("--max_pasdata_lendiff", default=-1, type=int)
    parser.add_argument("--max_bandwidth", default=4, type=int)
    # parser.add_argument("--resample_target_data", action="store_true")
    # parser.add_argument("--pasloss_coefficient", default=-1.0, type=float)
    # parser.add_argument("--otloss_coefficient", default=-1, type=float)
    # parser.add_argument("--supervised_mode", action="store_true")
    parser.add_argument("--pas_subsample_count", type=int, default=-1)
    parser.add_argument("--alignment_coefficient", type=float, default=-1)
    parser.add_argument("--adversarial_coefficient", type=float, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("--run_ordinary_validation", action="store_true")
    parser.add_argument("--flag_use_spacy_preprocessing", action="store_true")
    parser.add_argument("--langstr", type=str)
    parser.add_argument("--use_ppl_loss", action="store_true")
    parser.add_argument("--span_repr_mode", type=str, default="bge-m3")
    parser.add_argument("--remark", type=str, default="none")
    parser.add_argument("--forbid_bn", action="store_true")
    parser.add_argument("--forbid_merged_nll", action="store_true")
    parser.add_argument("--use_normalized_term_mlp", action="store_true")
    parser.add_argument("--use_onesided", action="store_true")
    parser.add_argument("--dropout", type=float, default=-1)
    parser.add_argument("--force_fp32", action="store_true")
    parser.add_argument("--max_length", type=int, default=40)
    # parser.add_argument('--eval_per_epoch', type=int, default=1)
    parser.add_argument("--wandb_tags", type=str, action="append")
    parser.add_argument("--val_check_interval", type=int, default=5000)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--flag_curriculum_learning", action="store_true")
    parser.add_argument("--flag_use_separate_nll_path", action="store_true")
    parser.add_argument("--unset_logppl", action="store_true")
    parser.add_argument("--unset_nll_weighing", action="store_true")
    parser.add_argument("--set_fast_model", action="store_true")
    parser.add_argument("--set_mode_offending_spans", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--unset_renormalizing_marginals", action="store_true")
    parser.add_argument("--set_lr", type=float, default=-1)
    parser.add_argument("--unset_bert_mode", action="store_true")
    parser.add_argument("--set_pas_suppression", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Spanoverlap-PCFG")
    parser.add_argument("--set_hit_count_threshold", type=int, default=-1)
    parser.add_argument("--preprocessing_pas_subsample_count", type=int, default=10000)
    parser.add_argument("--set_add_sentence_level_span", action="store_true")
    parser.add_argument("--set_spancomp_coefficient", type=float, default=-1)
    parser.add_argument("--flag_compute_relative_frequency", action="store_true")
    parser.add_argument("--set_training_mode", type=str, default=None)
    parser.add_argument("--set_min_span_reward", type=float, default=-100000)
    parser.add_argument("--distinct_prompt", action="store_true")
    parser.add_argument("--vocab_max_size", type=int, default=10000)
    parser.add_argument("--vocab_min_freq", type=int, default=5)
    parser.add_argument("--mode_reward", type=str, default=None)
    parser.add_argument("--set_reward_normalization", action="store_true")
    parser.add_argument("--unset_preprocessing_spacy", action="store_true")
    parser.add_argument("--unset_ptb_mode", action="store_true")
    parser.add_argument("--set_include_unary", action="store_true")
    parser.add_argument("--set_sample_epsilon", type=float, default=-1)
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--set_treecrf_samples", type=int, default=-1)
    parser.add_argument("--set_mode_reward", type=str, default=None)
    parser.add_argument("--set_flag_use_pos_unks", action="store_true")
    parser.add_argument("--run_val_only", action="store_true")
    parser.add_argument("--log_tree_path", type=str, default=None)
    parser.add_argument("--analysis_mode", action="store_true")
    parser.add_argument("--corr_mode", action="store_true")
    parser.add_argument("--output_detailed", action="store_true")
    parser.add_argument("--set_supervised_mode", action="store_true")
    parser.add_argument("--detailed_corr_mode", action="store_true")
    parser.add_argument("--unset_baseline", action="store_true")
    parser.add_argument("--use_pcfg_samples", action="store_true")
    parser.add_argument("--set_normalize_pcfg_conditional_logprob", action="store_true")
    # parser.add_argument("--batch_size", type=int, default=4)
    

    args2 = parser.parse_args() if force_args == [] else parser.parse_args(force_args)
    # torch.backends.cuda.matmul.allow_tf32 = Tr0ue
    if args2.force_fp32:
        torch.set_float32_matmul_precision("highest")
    else:
        torch.set_float32_matmul_precision("high")
    print(args2)

    yaml_cfg = yaml.safe_load(open(args2.conf, "r"))
    args = edict(yaml_cfg)
    args.update(args2.__dict__)
    args.ngpus = torch.cuda.device_count()

    if args.pas_subsample_count > -1:
        args.experimental.pas_subsample_count = args.pas_subsample_count
    if args.max_pasdata_lendiff > -1:
        args.experimental.max_pasdata_lendiff = args.max_pasdata_lendiff
    if args.alignment_coefficient > -1:
        args.experimental.alignment_coefficient = args.alignment_coefficient
    if args.adversarial_coefficient > -1:
        args.experimental.adversarial_coefficient = args.adversarial_coefficient

    if args.forbid_bn:
        args.model.use_bn = False

    if args.forbid_merged_nll:
        args.experimental.use_merged_nll = False

    if args.use_normalized_term_mlp:
        args.model.use_normalized_term_mlp = True
    else:
        args.model.use_normalized_term_mlp = False

    if args.dropout > -1:
        args.experimental.dropout = args.dropout

    if args.flag_curriculum_learning:
        args.experimental.flag_curriculum_learning = True

    if args.flag_use_separate_nll_path:
        args.experimental.flag_use_separate_nll_path = True

    if args.unset_logppl:
        args.experimental.flag_use_logppl = False

    if args.unset_nll_weighing:
        args.experimental.weigh_nll_loss = False

    if args.set_fast_model:
        args.model.use_fast_pcfg = True

    if args.set_mode_offending_spans:
        args.experimental.mode_offending_spans = True

    if args.unset_renormalizing_marginals:
        args.experimental.renormalizing_marginals = False

    if args.set_lr > 0:
        args.optimizer.lr = args.set_lr

    if args.batch_size > -1:
        args.train.batch_size = args.batch_size

    if args.unset_bert_mode:
        args.model.bert_mode = "disabled"

    if args.set_pas_suppression:
        args.experimental.suppress_pas_contrib = True

    if args.set_hit_count_threshold > -1:
        args.experimental.hit_count_threshold = args.set_hit_count_threshold

    if args.set_add_sentence_level_span:
        args.experimental.add_sentence_level_span = True

    if args.set_spancomp_coefficient > -1:
        args.experimental.spancomp_loss_weight = args.set_spancomp_coefficient

    if args.set_training_mode is not None:
        args.experimental.mode = args.set_training_mode

    if args.set_min_span_reward > -100000:
        args.experimental.min_span_reward = args.set_min_span_reward

    if args.mode_reward is not None:
        args.experimental.mode_reward = args.mode_reward

    if args.set_reward_normalization:
        args.experimental.reward_normalization = True

    if args.unset_preprocessing_spacy:
        args.preprocessing_use_spacy = False
    else:
        args.preprocessing_use_spacy = True

    if args.unset_ptb_mode:
        args.allow_ptb_eval = False
    else:
        args.allow_ptb_eval = True

    if args.set_include_unary:
        args.experimental.include_unary = True

    if args.set_sample_epsilon > -1:
        args.experimental.sample_epsilon = args.set_sample_epsilon

    if args.set_treecrf_samples > -1:
        args.experimental.num_samples = args.set_treecrf_samples

    if args.set_mode_reward is not None:
        args.experimental.mode_reward = args.set_mode_reward

    if args.set_flag_use_pos_unks:
        args.experimental.flag_use_pos_unks = True

    if args.set_supervised_mode:
        args.experimental.supervised_mode = True
    else:
        args.experimental.supervised_mode = False

    if args.unset_baseline:
        args.experimental.apply_mean_baseline = False

    if args.use_pcfg_samples:
        args.experimental.sample_mode = "pcfg"
    else:
        args.experimental.sample_mode = "crf"

    if args.set_normalize_pcfg_conditional_logprob:
        args.experimental.normalize_pcfg_conditional_logprob = True


    # if args.pas_length_limit > -1:
    #     args.experimental.pas_length_limit = args.pas_length_limit

    # device = f"cuda:{args.ngpus-1}" if torch.cuda.is_available() else "cpu"
    
    # if args.merge_pas_data:
    # args.train.batch_size *= 8
    device = "cuda"

    return args, device
