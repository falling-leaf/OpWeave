import argparse
import json
import re
from transformers import AutoTokenizer

MODEL_PATH = "/home/why/dbagent/pgdl/test/morphingdb_test/models/deberta-v3-large-squad2"
_TOKENIZER = None

def clean_control_chars(text):
    """
    使用正则表达式移除所有不可见/控制字符 (Unicode 类别 C)
    这将清除像 \u001a, \u0013, \x00 等干扰项
    """
    # 匹配所有 Unicode 控制字符
    return re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        _TOKENIZER = tokenizer
    return _TOKENIZER


def decode_ids(ids_arg: str):
    tokenizer = get_tokenizer()

    try:
        segments = [s.strip() for s in ids_arg.split(";") if s.strip()]
        token_id_lists = [[int(i.strip()) for i in seg.split(",") if i.strip()] for seg in segments]
        if not token_id_lists:
            return ""
    except ValueError:
        return json.dumps({"error": "Invalid input format"}, indent=4)

    encoded_inputs = tokenizer.pad(
        {"input_ids": token_id_lists},
        padding=True,
        return_tensors="pt"
    )

    raw_decoded_texts = tokenizer.batch_decode(
        encoded_inputs["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    output_data = {}
    for i, text in enumerate(raw_decoded_texts):
        clean_text = clean_control_chars(text).strip()
        output_data[f"paragraph_{i+1}"] = clean_text

    return json.dumps(output_data, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Batch decode token IDs to clean JSON.")
    parser.add_argument("--ids", type=str, required=True)
    args = parser.parse_args()
    print(decode_ids(args.ids))

if __name__ == "__main__":
    main()
