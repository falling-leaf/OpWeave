"""Long-lived tokenizer server for reader_decoder.

Loading the deberta tokenizer costs about 2.7s per process, and the reasoning
path calls it once per sample. A fresh process per sample spends its time in
the interpreter startup, which does not respond to CPU contention, so the
HotpotQA load levels come out flat. This keeps one process alive and decodes
each request in it, so the decode work itself is what the load contends with.

Protocol: one JSON object per line on stdin, {"ids": "..."} ; one JSON object
per line on stdout, {"text": "..."} or {"error": "..."}.
"""
import json
import sys

import reader_decoder


def main():
    reader_decoder.get_tokenizer()
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ids_arg = json.loads(line).get("ids", "")
            text = reader_decoder.decode_ids(ids_arg)
        except Exception as exc:
            text = json.dumps({"error": str(exc)})
        sys.stdout.write(json.dumps({"text": text}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
