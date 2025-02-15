# %%
import sys
import os
import json


path = sys.argv[1] #"chinese/ptb_zh-full.gd_instruction.batch/val.query"
for section in ["train", "val", "test"]:
    fn = os.path.join(path, section+".query")
    batch_obj_list = []
    with open(fn+".batch_obj") as f:
        for line in f:
            batch_obj_list.append(line.strip())
        
    # %%
    import re
    def extract_sent_from_numbered_list(raw):
        # print("incoming raw", raw)
        found = re.findall("\n\d+\.([^\n]+)", raw)
        return [i.strip() for i in found]

    # %%
    from openai import OpenAI
    
    with open("openai_key", "rb") as f:
        openai_params = json.load(f)
        
    client = OpenAI(
        **openai_params
    )


    output_file_id_list = []
    for batch_obj in batch_obj_list:
        print(batch_obj)
        print(client.batches.retrieve(batch_obj))
        output_file_id_list.append(client.batches.retrieve(batch_obj).output_file_id)
    # client.batches.retrieve("batch_abc123")

    # %%
    output_file_id_list[0]

    # %%


    # %%
    import json

    # %%


    # %%
    n = 0
    from tqdm import tqdm
    # openai_output = [{"instructions":[], "paraphrases": [], "logprobs":[]} for i in range(10000)]

    openai_output = [{"instructions":[], "paraphrases": [], "logprobs":[]} for i in range(100000)]
    max_sent_id = -1

    print(output_file_id_list)
    for out_fid in output_file_id_list:
        print("retrieving content from", out_fid)
        content = client.files.content(out_fid)
        outit = content.iter_lines()
        for line in tqdm(outit):
            out = json.loads(line)
            custom_id = out["custom_id"].split("|SEP|")
            partition, dst_id = custom_id[0].split("-")
            # if 'question' in  custom_id[1]: continue # skip questions
            dst_id = int(dst_id)
            if dst_id>=max_sent_id:
                max_sent_id = dst_id
            instruction = custom_id[1]
            n+=1
            # print(dst_id, instruction)
            samples = out["response"]["body"]["choices"][0]["message"]["content"]
            logprobs = out["response"]["body"]["choices"][0]["logprobs"]
            # print(samples)
            # print(samples)
            # print(extract_sent_from_numbered_list("\n" + samples + "\n"))
            # print(logprobs)
            try:
                samples  = json.loads(samples)
                sudo_paraphrases = samples['paraphrases']
                paraphrases = []
                for p in sudo_paraphrases:
                    if not type(p) is str:
                        continue
                    if "[SEP]" in p:
                        # continue
                        p1, p2 = p.split("[SEP]")
                        # paraphrases.append(p1.strip())
                        paraphrases.append(p2.strip())
                    else:
                        paraphrases.append(p)
                all_paraphrases_string = [type(w) is str for w in paraphrases]
                # print(all_paraphrases_string)
                if not all(all_paraphrases_string):
                    raise Exception("Not all paraphrases are strings")
            except Exception as e:
                print(samples)
                paraphrases = []
                pass
            # print(paraphrases)
            openai_output[dst_id]["instructions"].extend([instruction]* len(paraphrases))
            openai_output[dst_id]["paraphrases"].extend(paraphrases)
            # openai_output[dst_id]["logprobs"].append(logprobs)
            # if n>18:
                # break

    # %%
    # openai_output = [o for o in openai_output if len(o["instructions"])>0]
    openai_output = openai_output[:max_sent_id+1]

    # %%
    import pickle


    with open(fn+".openai_output", "wb") as f:
        pickle.dump(openai_output, f)

# %%


# %%
