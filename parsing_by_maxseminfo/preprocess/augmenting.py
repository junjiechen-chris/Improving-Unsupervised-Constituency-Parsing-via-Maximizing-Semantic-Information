
from nltk import Tree
import argparse
import pickle
import os
from parsing_by_maxseminfo.utils.GPTEnhancer import GPTEnhancer
from parsing_by_maxseminfo.utils.utils import factorize
from parsing_by_maxseminfo.utils.constants import nonword_pos
from openai import OpenAI
import json

with open("openai_key", "rb") as f:
    openai_params = json.load(f)
    
client = OpenAI(
    **openai_params
)

def build_augmentation_queries(file_name, dst_label, language, max_seqlen=100, prompt_keys=[], dst_truncate=100000):
    dst_queries = []

    def process(line, dst_id):
        tree = Tree.fromstring(line)

        token = tree.pos()
        word, pos = zip(*token)

        gold_tree = factorize(tree)
        mask_isword = [p not in nonword_pos for p in pos]
        if sum(mask_isword) > max_seqlen:
            return None, None, None, None, None, None, None, None
        

        queries = []
        # samples_prompt = []
        sentence_joiner = ' ' if not language=='chinese' else ''
        print("orig: ", sentence_joiner.join(word))
        for k in prompt_keys:
            # try:
            query = gpt_model.augment_prep_josn(
                sentence_joiner.join(word),
                id=f"{dst_label}-{dst_id}",
                prompt_key=k,
                flag_fixing_unicode=args.fix_unicode,
                flag_json_mode=args.json_mode,
                language=language,
            )
            queries.append(query)


        return [queries] 

    with open(file_name, "r") as f:
        lines = f.read().splitlines()
        # with ThreadPool(16) as pool:
        result = map(process, lines[:dst_truncate], range(dst_truncate))
        for pack in result:
            if pack[0] is not None:
                query = pack[0]
                dst_queries.extend(query)

    return {
        "dst_queries": dst_queries,
    }


def create_dataset(file_name, fn_openai_output, dst_label, max_seqlen=100, prompt_keys=[], dst_truncate=100000):
    word_array = []
    pos_array = []
    gold_trees = []
    seqlen_array = []
    pas_sample_array = []
    pas_prompt_key_array = []

    def process(line, openai_output, dst_id):
        tree = Tree.fromstring(line)

        token = tree.pos()
        word, pos = zip(*token)

        gold_tree = factorize(tree)
        mask_isword = [p not in nonword_pos for p in pos]
        if sum(mask_isword) > max_seqlen:
            return None, None, None, None, None, None, None, None
        print("orig: ", " ".join(word))

        paraphrases = openai_output["paraphrases"]
        prompts = openai_output["instructions"]



        return word, pos, gold_tree, paraphrases, prompts
        # return [queries] 

    with open(file_name, "r") as f:
        lines = f.read().splitlines()
    with open(fn_openai_output, "rb") as f:
        openai_outputs = pickle.loads(f.read())

    result = map(process, lines[:dst_truncate], openai_outputs[:dst_truncate], range(dst_truncate))
    for pack in result:
        if pack[0] is not None:
            (
                word,
                pos,
                gold_tree,
                sample,
                samples_prompt,
            ) = pack
            word_array.append(word)
            pos_array.append(pos)
            gold_trees.append(gold_tree)
            pas_sample_array.append(sample)
            pas_prompt_key_array.append(samples_prompt)

    return {
        "word": word_array,
        "pos": pos_array,
        "gold_tree": gold_trees,
        "seq_len": seqlen_array,
        "pas_sample": pas_sample_array,
        "pas_prompt_key": pas_prompt_key_array,
    }

def upload_and_launch_batch(file_name, queries, batch_size=45000):

    num_batches = len(queries) // batch_size + 1
    batch_obj_list = []
    for i in range(num_batches):
        path = os.path.join(args.cache_path, f"{file_name}{i}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for q in queries[i*batch_size:(i+1)*batch_size]:
                f.write(json.dumps(q))
                f.write("\n")
        batch_input = gpt_model.upload_batch(path)
        batch_input_file_id = batch_input.id

        batch_obj = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
            }
        )
        batch_obj_list.append(batch_obj)
    
    with open(os.path.join(args.cache_path, f"{file_name}.batch_obj"), "w") as f:
        for batch_obj in batch_obj_list:
            f.write(batch_obj.id)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess ptb file.")
    parser.add_argument("--train_file")
    parser.add_argument("--train_openai_output", default=None)
    parser.add_argument("--val_file")
    parser.add_argument("--val_openai_output", default=None)
    parser.add_argument("--test_file")
    parser.add_argument("--test_openai_output", default=None)
    parser.add_argument("--cache_path")
    parser.add_argument("--max_seqlen", default=100000, type=int)
    parser.add_argument("--gpt_modelstr", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--dst_truncate", default=10, type=int)
    parser.add_argument("--prompt_key", nargs="+")
    parser.add_argument("--fix_unicode", action="store_true")
    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--language", type=str)


    args = parser.parse_args()
    gpt_model = GPTEnhancer(openai_params, args.gpt_modelstr)

    from pathlib import Path
    import json
    
    c_merge = args.train_openai_output is None

    Path(args.cache_path).mkdir(parents=True, exist_ok=True)
    if not c_merge:
        train_queries = create_dataset(
            args.train_file,
            args.train_openai_output,
            dst_label="train",
            max_seqlen=args.max_seqlen,
            prompt_keys=args.prompt_key,
            dst_truncate=args.dst_truncate,
        )
        with open(os.path.join(args.cache_path, "train.pickle"), "wb") as f:
            pickle.dump(train_queries, f)
    else:
        train_queries = build_augmentation_queries(
            args.train_file,
            dst_label="train",
            max_seqlen=args.max_seqlen,
            prompt_keys=args.prompt_key,
            dst_truncate=args.dst_truncate,
            language=args.language,
        )
        upload_and_launch_batch("train.query", train_queries["dst_queries"])
        
    
    # upload_and_launch_batch("train.query", train_queries["dst_queries"])

    if not c_merge:
        dev_queries = create_dataset(
            args.val_file,
            args.val_openai_output,
            dst_label="val",
            max_seqlen=args.max_seqlen,
            prompt_keys=args.prompt_key,
            dst_truncate=args.dst_truncate,
        )
        with open(os.path.join(args.cache_path, "val.pickle"), "wb") as f:
            pickle.dump(dev_queries, f)
    else:
        dev_queries = build_augmentation_queries(
            args.val_file,
            dst_label="val",
            max_seqlen=args.max_seqlen,
            prompt_keys=args.prompt_key,
            dst_truncate=args.dst_truncate,
            language=args.language,
        )
        upload_and_launch_batch("val.query", dev_queries["dst_queries"])

    if not c_merge:
        test_queries = create_dataset(
            args.test_file,
            args.test_openai_output,
            dst_label="test",
            max_seqlen=args.max_seqlen,
            prompt_keys=args.prompt_key,
            dst_truncate=args.dst_truncate,
        )
        with open(os.path.join(args.cache_path, "test.pickle"), "wb") as f:
            pickle.dump(test_queries, f)
    else:
        test_queries = build_augmentation_queries(
            args.test_file,
            dst_label="test",
            max_seqlen=args.max_seqlen,
            prompt_keys=args.prompt_key,
            dst_truncate=args.dst_truncate,
            language=args.language,
        )
        upload_and_launch_batch("test.query", test_queries["dst_queries"])
