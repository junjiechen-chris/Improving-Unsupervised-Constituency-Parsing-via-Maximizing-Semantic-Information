from ast import arg
from re import T
import lightning.pytorch as L
import numpy as np
from parsing_by_maxseminfo.parser.model.C_PCFG import  CompoundPCFGFixedCostReward 
from parsing_by_maxseminfo.parser.model.SC_PCFG import SCPCFGFidxedCostReward
from parsing_by_maxseminfo.parser.model.SN_PCFG import  SNPCFGFixedCostReward, SNPCFGFixedCostRewardA2C
import torch
from parsing_by_maxseminfo.parser.helper.metric import (
    UF1,
    UAS,
    LikelihoodMetric,
    MetricAccumulator,
    RewardAccumulator,
)
from parsing_by_maxseminfo.parser.model.N_PCFG import NeuralPCFGFixedCostReward, NeuralPCFGFixedCostRewardA2C
from parsing_by_maxseminfo.parser.model.TN_PCFG import  TNPCFGFixedCostReward


class LitXNPCFGFixedCost(L.LightningModule):
    def __init__(
        self,
        basemodel,
        model_params,
        vocab_size,
        experimental_config,
        optim_config,
        langstr,    
    ):
        super().__init__()

        print(
            f"Constructing LitNPCFGFixedCost with experimental config {experimental_config}"
        )
        self.config = experimental_config
        self.optim_config = optim_config
        self.model = None
        self.validation_step_outputs = []
        self.flag_use_ppl_as_nll_loss = False
        self.test_mode = "spacy"

        self.metric_f1 = UF1()
        self.metric_ll = LikelihoodMetric()

        self.save_hyperparameters()
        # self.automatic_optimization = False

    def print_base(self):
        for base in self.__class__.__bases__:
            print(base.__name__)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses must implement this method.")
        

    def validation_step(self, batch, batch_idx):
        # from parser.helper.metric import uf1_tpfpfn

        x, y = batch

        x_target = {"word": x["target_id_array"], "seq_len": x["target_len_array"]}
        outputs = self.model.evaluate(x_target, decode_type="mbr", span_mask=None)
        num_words = x["target_len_array"]
        ll = outputs["partition"]

        self.metric_f1(outputs["prediction"], y["gold_tree"])
        self.metric_ll(ll, num_words)

        return {
            # "tpfpfn": tpfpfn,
            "ll_loss": ll.cpu(),
        }

    def test_step(self, batch, batch_idx):
        x, y = batch

        x_target = {"word": x["target_id_array"], "seq_len": x["target_len_array"]}
        outputs = self.model.evaluate(
            x_target,
            decode_type="mbr",
            span_mask=(
                x["tree_mask"].to(self.device) if x["tree_mask"] is not None else None
            ),
        )
        num_words = x["target_len_array"]
        ll = outputs["partition"]

        pred = outputs["prediction"]
        gold = y["gold_tree"]

        self.metric_f1(gold, pred)
        self.metric_ll(ll, num_words)

        return {
            "ll_loss": ll.cpu(),
        }

    def on_test_epoch_start(self):
        self.model.train(False)
        self.metric_f1 = UF1()
        self.metric_ll = LikelihoodMetric()

    def on_test_epoch_end(self):
        self.model.train(True)
        sf1 = self.metric_f1.sentence_uf1
        cf1 = self.metric_f1.corpus_uf1
        ll = self.metric_ll.avg_likelihood
        ppl = self.metric_ll.perplexity

        self.log_dict(
            {
                "test/corpus_f1": cf1,
                "test/sentence_f1": sf1,
                "test/avg_ll": ll,
                "test/avg_ppl": ppl,
            }
        )

        self.metric_f1 = UF1()
        self.metric_ll = LikelihoodMetric()

        return {
            "corpus_f1": cf1,
            "sentence_f1": sf1,
            "avg_ll": ll,
            "avg_ppl": ppl,
        }

    def on_validation_epoch_start(self):
        self.metric_f1 = UF1()
        self.metric_ll = LikelihoodMetric()

    def on_validation_epoch_end(self):

        sf1 = self.metric_f1.sentence_uf1
        cf1 = self.metric_f1.corpus_uf1
        ll = self.metric_ll.avg_likelihood
        ppl = self.metric_ll.perplexity

        self.log_dict(
            {
                "val/corpus_f1": cf1,
                "val/sentence_f1": sf1,
                "val/avg_ll": ll,
                "val/avg_ppl": ppl,
            }
        )

        self.metric_f1 = UF1()
        self.metric_ll = LikelihoodMetric()

        return {
            "corpus_f1": cf1,
            "sentence_f1": sf1,
            "avg_ll": ll,
            "avg_ppl": ppl,
        }

    def configure_optimizers(self):
        optim_params = self.optim_config
        if optim_params.name == "adamw":
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=optim_params.lr,
                betas=(optim_params.mu, optim_params.nu),
                weight_decay=optim_params.weight_decay,
            )
        elif optim_params.name == "adam":
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optim_params.lr,
                betas=(optim_params.mu, optim_params.nu),
                weight_decay=(
                    optim_params.weight_decay
                    if hasattr(optim_params, "weight_decay")
                    else 0.0
                ),
            )
        return optimizer

    def set_test_mode(self, mode):
        assert mode in [ "spacy"]
        self.test_mode = mode


class LitXNPCFGFCReward(LitXNPCFGFixedCost):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        basemodel = args[0]
        if basemodel == "NPCFG":
            self.model = NeuralPCFGFixedCostReward(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        elif basemodel == "TNPCFG":
            self.model = TNPCFGFixedCostReward(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        elif basemodel == "CPCFG":
            self.model = CompoundPCFGFixedCostReward(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        elif basemodel == "SNPCFG":
            self.model = SNPCFGFixedCostReward(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        elif basemodel == "SNPCFGA2C":
            self.model = SNPCFGFixedCostRewardA2C(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        elif basemodel == "NPCFGA2C":
            self.model = NeuralPCFGFixedCostRewardA2C(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        elif basemodel == "SCPCFG":
            self.model = SCPCFGFidxedCostReward(
                args[1], args[2], span_repr_mode="em", langstr=args[5]
            )
        else:
            raise NotImplementedError(f"{basemodel} is not allowed")

        self.model_params = args[1]

        self.log_tree_path = None
        self.output_detailed = False

        self.rl_coeff = 0.0
        self.maxent_coeff = 0.0


    def training_step(self, batch, batch_idx):
        x, y = batch
        x_target = {
            "word": x["target_id_array"],
            "seq_len": x["target_len_array"],
            "form": x["target_form_array"],
            "reward": x["reward"],
            "encoded_input": x["encoded_input"],
            "pas_indicator": x["pas_indicator"],
            "gold_tree_mask": x["gold_tree_mask"],
        }

        outputs = self.model.loss(
            x_target,
            run_wd=True if self.config.alignment_coefficient > 0 else False,
            mode=self.config.mode,
            dropout=self.config.dropout if hasattr(self.config, "dropout") else 0.0,
            rl_coeff=self.rl_coeff,
            maxent_coeff=self.maxent_coeff,
            sample_epsilon=(
                self.config.sample_epsilon
                if hasattr(self.config, "sample_epsilon")
                else 0.0
            ),
            include_unary=(
                self.config.include_unary
                if hasattr(self.config, "include_unary")
                else False
            ),
            num_samples=(
                self.config.num_samples if hasattr(self.config, "num_samples") else 4
            ),
            supervised_mode=(
                self.config.supervised_mode
                if hasattr(self.config, "supervised_mode")
                else False
            ),
            sample_mode = self.config.sample_mode,
        )

        loss = outputs["loss"]
        stat_nll = outputs["stat_nll"]
        stat_logppl = outputs["stat_logppl"]
        stat_reward = outputs["stat_reward"]
        stat_bl_reward = outputs["stat_baseline_reward"]
        stat_entropy = outputs["stat_entropy"]
        stat_rw = outputs["stat_rw"]

        coeff_rl = outputs["coeff_rl"]
        coeff_maxent = outputs["coeff_maxent"]


        self.log("train/train_loss", loss.cpu().item(), on_step=True, on_epoch=False)
        self.log("train/nll", stat_nll.cpu().item(), on_step=True, on_epoch=False)
        self.log("train/logppl", stat_logppl.cpu().item(), on_step=True, on_epoch=False)
        self.log(
            "train/sample_reward",
            stat_reward.cpu().item(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/baseline_reward",
            stat_bl_reward.cpu().item(),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            "train/entropy", stat_entropy.cpu().item(), on_step=True, on_epoch=False
        )
        self.log("train/rl_coeff", coeff_rl, on_step=True, on_epoch=False)
        self.log("train/maxent_coeff", coeff_maxent, on_step=True, on_epoch=False)
        self.log(
            "train/advantage std", stat_rw.cpu().item(), on_step=True, on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        # from parser.helper.metric import uf1_tpfpfn
        import json

        x, y = batch
        self.dst_size += len(x["target_id_array"])

        x_target = {
            "word": x["target_id_array"],
            "seq_len": x["target_len_array"],
            "form": x["target_form_array"],
            "reward": x["reward"],
            "competing_span_mask": x["competing_span_mask"],
            "encoded_input": x["encoded_input"],
            "pas_indicator": x["pas_indicator"],
        }
        outputs = self.model.evaluate(x_target, decode_type="mbr", span_mask=None)
        num_words = x["target_len_array"]
        ll = outputs["partition"]

        self.metric_reward(outputs["prediction"], x["reward"])

        self.metric_f1(outputs["prediction"], y["gold_tree"], log_metric=None)
        self.metric_ll(ll, num_words)

        return {
            # "tpfpfn": tpfpfn,
            "ll_loss": ll.cpu(),
        }

    def on_validation_epoch_start(self):
        self.model.train(False)
        self.dst_size = 0
        self.metric_f1 = UF1(log_file=self.log_tree_path)
        self.metric_ll = LikelihoodMetric()
        self.metric_reward = RewardAccumulator()

    def on_validation_epoch_end(self):

        self.model.train(True)
        sf1 = self.metric_f1.sentence_uf1
        cf1 = self.metric_f1.corpus_uf1
        ll = self.metric_ll.avg_likelihood
        ppl = self.metric_ll.perplexity

        reward = self.metric_reward.average_reward
        print("dst_size", self.dst_size)

        # spannll = self.metric_spannll.average
        # spancomp = self.metric_spancomp.average

        self.log_dict(
            {
                "val/corpus_f1": cf1,
                "val/sentence_f1": sf1,
                "val/avg_ll": ll,
                "val/avg_ppl": ppl,
                "val/avg_reward": reward,
                # "val/spannll": spannll,
                # "val/spancomp": spancomp,
            }
        )

        self.metric_f1 = UF1()
        self.metric_ll = LikelihoodMetric()

        return {
            "corpus_f1": cf1,
            "sentence_f1": sf1,
            "avg_ll": ll,
            "avg_ppl": ppl,
        }
