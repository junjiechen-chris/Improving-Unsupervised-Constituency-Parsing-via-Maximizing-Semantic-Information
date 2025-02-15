from parsing_by_maxseminfo.utils.constants import prompt, output_format
from openai import OpenAI
from parsing_by_maxseminfo.utils.utils import extract_sent_from_numbered_list


class CLMEnhancer:
    def __init__(self) -> None:
        pass
    


class GPTEnhancer(CLMEnhancer):
    def __init__(self, openai_params, model_str) -> None:
        super().__init__()

        self.client = OpenAI(
            **openai_params
        )
        self.model_str = model_str
        self.prompt_map = prompt

    def augment_prep_josn(
        self,
        sentence,
        id,
        language,
        prompt_key,
        flag_fixing_unicode=False,
        flag_json_mode=False,
    ):
        import hashlib

        assert flag_json_mode, "flag_json_mode must be True"
        if "question" in prompt_key:
            generation_format = output_format["json_qa"]
        else:
            generation_format = output_format["json_sent"]

        def get_sha256_hash(text):
            return hashlib.sha256(text.encode()).hexdigest()

        message = {
            "custom_id": f"{id}|SEP|{prompt_key}|SEP|{get_sha256_hash(sentence)}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": f"{self.model_str}",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are ChatGPT, a large language model trained by OpenAI. Carefully heed the user's instructions.",
                    },
                    {
                        "role": "user",
                        "content": self.prompt_map[prompt_key].format(
                            output_format=generation_format,
                            sentence=sentence,
                            lang=language,
                        ),
                    },
                ],
                "max_tokens": 1000,
            },
        }
        if flag_json_mode:
            message["body"]["response_format"] = {"type": "json_object"}
        return message

    def upload_batch(self, fn):
        batch_input_file = self.client.files.create(
            file=open(fn, "rb"),
            purpose="batch",
        )
        return batch_input_file

    def augment(
        self, sentence, prompt_key="reorder_diverse", flag_fixing_unicode=False
    ):
        raise NotImplementedError("augment function is not allowed in batched augmentation mode")
    
    
    def augment_with_infix(self, sentence, infix, prompt_key="reorder_diverse_infix"):
        response = self.client.chat.completions.create(
            model=self.model_str,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt_map[prompt_key].format(
                        sentence=sentence, infix=infix
                    ),
                }
            ],
            logprobs=True,
            max_tokens=192,
            temperature=1,
            n=1,
        )

        response_literal = ""
        for tok in response.choices[0].logprobs.content:
            response_literal += tok.token
        # print(paraphrased[i])
        returned_samples = extract_sent_from_numbered_list(
            "\n" + response_literal + "\n"
        )
        return returned_samples