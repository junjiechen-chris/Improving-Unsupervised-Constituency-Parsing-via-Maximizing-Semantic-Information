This is a preliminary code release for **Improving Unsupervised Constituency Parsing via Maximizing Semantic Information**. 

The training and evaluation can be run using [this colab notebook](https://drive.google.com/file/d/1RYPwPp8aEJ7-gjgyxJRVYUKEZWP1JbOW/view?usp=sharing)

Detailed usage of this codebase will be detailed in later days.

Pre-processed datasets are available in [huggingface](https://huggingface.co/datasets/HarpySeal/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information/tree/main)

----
### How to run the code
#### Installation
```
git clone https://github.com/junjiechen-chris/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information.git
pip install -e Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information
cp -r Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information/config config
```
#### Preparing Data
Option 1: Download preprocessed data from huggingface repo
```
!mkdir -p data
!wget https://huggingface.co/datasets/HarpySeal/Improving-Unsupervised-Constituency-Parsing-via-Maximizing-Semantic-Information/resolve/main/english.zip
!unzip -o english.zip -d data/english
```

Option 2: Proprocess the data with scripts. 

We will update the code later

#### Running training code
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



