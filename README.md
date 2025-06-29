This is a preliminary code release for [Improving Unsupervised Constituency Parsing via Maximizing Semantic Information](https://openreview.net/forum?id=qyU5s4fzLg&noteId=qyU5s4fzLg) (Spotlight @ ICLR25).  

The training and evaluation can be run using [this colab notebook](https://drive.google.com/file/d/1RYPwPp8aEJ7-gjgyxJRVYUKEZWP1JbOW/view?usp=sharing)

More details in this [slide deck](https://s3.g.s4.mega.io/aczeau2wd2mrkhulcbykdz5ujnqvr2zinc7jm/homepage-assets/ICLR-Slide.pdf)

----
## How to run the code
### Installation
```
git clone https://github.com/junjiechen-chris/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information.git
pip install -e Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information
cp -r Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information/config config
```
### Preparing Data
Option 1: Download preprocessed data from huggingface repo.
All datasets are available in [huggingface](https://huggingface.co/datasets/HarpySeal/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information/tree/main)

```
!mkdir -p data
!wget https://huggingface.co/datasets/HarpySeal/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information/resolve/main/english.zip
!unzip -o english.zip -d data/english
```

Option 2: Proprocess the data with scripts. 
1. Prepare the bracketed treebank files (e.g., [PTB validation data](https://github.com/nikitakit/self-attentive-parser/blob/master/data/22.auto.clean)) and generate the paraphrases using OpenAI's batch API. Filling the OpenAI key information in the `openai_key` file. The estimated cost of generating paraphrases with the `GPT-4o-mini` model is around 5 USD.
``` 
python -m parsing_by_maxseminfo.preprocess.augmenting \
	--train_file $(CORPUS_DIR)/ptb-train.txt \
	--val_file $(CORPUS_DIR)/ptb-val.txt \
	--test_file $(CORPUS_DIR)/ptb-test.txt \
	--cache_path $(WORKDIR) \
	--gpt_modelstr gpt-4o-mini \
	--json_mode \
	--dst_truncate 1000000 \
	--language english \
	--prompt_key shuffling_gd_en_ew passive_gd_en_ew active_gd_en_ew clefting_gd_en_ew wh_question_gd_en_ew confirmatory_question_gd_en_ew topicalization_gd_en_ew heavynp_shift_gd_en_ew

```
2. Download the generated paraphrases after the generation.
```
python -m parsing_by_maxseminfo.preprocess.downloading_from_openai $(WORKDIR)

python -m parsing_by_maxseminfo.preprocess.augmenting \
	--train_file $(CORPUS_DIR)/ptb-train.txt \
	--val_file $(CORPUS_DIR)/ptb-val.txt \
	--test_file $(CORPUS_DIR)/ptb-test.txt \
	--train_openai_output $(WORKDIR)/train.query.openai_output \
	--val_openai_output $(WORKDIR)/val.query.openai_output \
	--test_openai_output $(WORKDIR)/test.query.openai_output \
	--cache_path $(WORKDIR) \
	--gpt_modelstr gpt-4o-mini \
	--json_mode \
	--dst_truncate 100000000000 \
	--language english \
	--prompt_key shuffling_gd_en_ew passive_gd_en_ew active_gd_en_ew clefting_gd_en_ew wh_question_gd_en_ew confirmatory_question_gd_en_ew topicalization_gd_en_ew heavynp_shift_gd_en_ew
```

3. Precompute the substring frequency and cache the preprocessed file. Make sure to change the dataset path in the config file
```
python -m parsing_by_maxseminfo.preprocess.caching -c config/pas-grammar/english-ew-reward-tbtok-idf/npcfg_nt60_t120_en.spacy-10k-merged-0pas-fast-6-3-rlstart0-newdata.yaml --ckpt_dir tmp --flag_compute_relative_frequency --unset_preprocessing_spacy
```

### Running training code
The below code train the PCFG model as described in the paper. The code provides several alternative training modes as listed below. Please choose one that fits your purpose.
- rl: SemInfo mean-baseline training with CRF as explained in the main text
- nll: LL training as explained in the main text
- a2c: Stepwise SemInfo training with CRF as explained in Appendix A1
- a2c_v0: Posterior V0 training with CRF
- ta2c_rules: Stepwise SemInfo training with PCFG
- ta2c: Posterior V0 training with PCFG
- tavg: Posterior mean-baseline training with PCFG
```
python -m parsing_by_maxseminfo.train -c config/pas-grammar/english-ew-reward-tbtok-idf/npcfg_nt60_t120_en.spacy-10k-merged-0pas-fast-6-3-rlstart0.yaml --max_length=40 --set_training_mode=rl --set_min_span_reward=-4 --unset_ptb_mode --ckpt_dir=./checkpoints/english/seminfo --set_mode_reward=log_tfidf --set_include_unary --langstr=english --remark english-rl --wandb_project PCFG-SemInfo &> log.english-rl
```



