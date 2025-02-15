
nonword_pos = ["``", "''", ":", "-RRB-", ",", ".", "-LRB-", "#", "$"]

output_format = {
    "json_sent": "The generated sentences should be put into a json object like {'paraphrases': [<sent1>, <sent2>, ...]}",
    "json_qa": "The generated sentences should be put into a json object like {'paraphrases': [<question1>[SEP]<answer1>, <question2>[SEP]<answer2>, ...]}.",
    "json_qa_v2": "The generated sentences should be put into a json object like {'paraphrases': [\{'question': <question1>, 'answer': <answer1>\}, ...]}.",
    "list_qa": "The generated sentences should be put into a numbered list like \n[01] <question1> [SEP] <answer1>\n[02] <question2> [SEP] <answer2>\n...",
    "list_sent": "The generated sentences should be put into a numbered list like \n[01] <sentence1>\n[02] <sentence2>\n...",
}

prompt = {
    "shuffling_gd_en_ew": "Create grammatical sentences by shuffling the phrases in the below sentence. The generated sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
    "tense_gd_en_ew": "Create grammatical sentences by changing the tense in the below sentence. The generated sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
    "passive_gd_en_ew": "Create grammatical sentences by restating the below sentences in passive voice. The generated sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
    "active_gd_en_ew": "Create grammatical sentences by restating the below sentences in active voice. The generated sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
    "clefting_gd_en_ew": "Create grammatical clefting sentences based on the below sentence. The generated sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
    "wh_question_gd_en_ew": "Create pairs of interrogative and its answers based on the below sentence. The generated sentences must be grammatically correct and be explicit. The sentences must be in {lang}. Use the same word as in the original sentence. The answer to the questions should be a substring of the given sentence. {output_format}\n{sentence}",
    "confirmatory_question_gd_en_ew": "Create pairs of confirmatory questions and its answers based on the below sentence. The generated sentences must be grammatically correct and textually diverse. The sentences must be in {lang}. Use the same word as in the original sentence. The answer to the questions should be a substring of the given sentence. {output_format}\n{sentence}",
    "topicalization_gd_en_ew": "Create grammatical sentences by performing the topicalization transformation to the below sentence. The sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
    "heavynp_shift_gd_en_ew": "Create grammatical sentences by performing the heavy NP shift transformation to the below sentence. The sentences must be in {lang}. Use the same word as in the original sentence. {output_format}\n{sentence}",
}
