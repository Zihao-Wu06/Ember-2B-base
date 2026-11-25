#!/usr/bin/env python
"""Generative benchmarks of an exported checkpoint on the Mac with incremental decoding (decode.py), greedy.

    .venv-mac/bin/python runs/mac_eval/gen_eval.py --tasks gsm8k,humaneval --gsm8k-limit 100 --out runs/mac_eval/g.json

gsm8k: 5-shot "Question: ...\\nAnswer: <worked solution> #### N" (lm-eval-harness format). The shots are the first 5
  test problems (only the test split is on disk), so problems 6.. are scored; strict = the number after "####",
  flexible = the last number in the output. Not directly comparable to the usual 8-shot numbers from the train split.
humaneval: the prompt as given, stops at a new top-level def / class / if / print / comment; the completion runs with
  the official tests in a separate `python -I` process (temp dir, 10 s limit). Completions that touch the file
  system, processes or imports beyond plain computation are not run and count as failures."""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mac_lm  # noqa: E402
import decode  # noqa: E402

BENCH = os.path.join(mac_lm.ROOT, "data", "raw", "bench")
UNSAFE = re.compile(r"\b(os|sys|subprocess|shutil|socket|pathlib|ctypes|importlib|signal|multiprocessing|threading)\b"
                    r"|__import__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|rmtree|unlink")
HE_STOPS = ("\ndef ", "\nclass ", "\nif ", "\nprint(", "\n#")


def read_jsonl(*parts):
    with open(os.path.join(BENCH, *parts), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _num(s):
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k(model, tok, limit, shots=5, max_new=320, log=None):
    data = read_jsonl("gsm8k", "test.jsonl")
    prefix = "".join(f"Question: {d['question']}\nAnswer: {d['answer']}\n\n" for d in data[:shots])
    cache = decode.prefill(model, mac_lm.encode(tok, prefix))
    strict = flex = 0
    test = data[shots:shots + limit]
    for i, d in enumerate(test):
        t0 = time.time()
        stop = lambda out: any(s in tok.decode(out) for s in ("\n\n", "Question:"))  # noqa: E731
        ids = decode.generate_ids(model, tok.encode(f"Question: {d['question']}\nAnswer:").ids, max_new, stop,
                                  init=cache)
        text = tok.decode(ids).split("\n\n")[0].split("Question:")[0]
        gold = _num(d["answer"].split("####")[-1])
        m = re.search(r"####\s*(-?[\d\.,\$]+)", text)
        s_ok = m is not None and _num(m.group(1)) == gold
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        f_ok = bool(nums) and _num(nums[-1]) == gold
        strict += s_ok
        flex += f_ok
        if log is not None:
            log.write(json.dumps({"task": "gsm8k", "i": shots + i, "gold": gold, "strict": s_ok, "flexible": f_ok,
                                  "output": text, "seconds": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
            log.flush()
        if (i + 1) % 10 == 0:
            print(f"  gsm8k {i + 1}/{len(test)}: strict {strict}, flexible {flex}", flush=True)
    return {"task": "gsm8k", "n": len(test), "shots": shots, "strict": strict / len(test), "flexible": flex / len(test)}


def run_program(code, timeout=10):
    with tempfile.TemporaryDirectory() as d:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", code], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin", "HOME": d})
            return r.returncode == 0, r.stderr[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


def humaneval(model, tok, limit, max_new=384, log=None):
    data = read_jsonl("humaneval", "openai_humaneval", "test-00000-of-00001.jsonl")[:limit]
    passed = skipped = 0
    for i, d in enumerate(data):
        t0 = time.time()

        def stop(out):
            return any(s in tok.decode(out) for s in HE_STOPS)
        ids = decode.generate_ids(model, mac_lm.encode(tok, d["prompt"]), max_new, stop)
        comp = tok.decode(ids)
        cut = min([comp.find(s) for s in HE_STOPS if s in comp] or [len(comp)])
        comp = comp[:cut]
        if UNSAFE.search(comp):
            ok, err = False, "not run (unsafe pattern)"
            skipped += 1
        else:
            ok, err = run_program(d["prompt"] + comp + "\n\n" + d["test"] + f"\n\ncheck({d['entry_point']})\n")
        passed += ok
        if log is not None:
            log.write(json.dumps({"task": "humaneval", "id": d["task_id"], "passed": ok, "completion": comp,
                                  "error": "" if ok else err, "seconds": round(time.time() - t0, 1)},
                                 ensure_ascii=False) + "\n")
            log.flush()
        if (i + 1) % 10 == 0:
            print(f"  humaneval {i + 1}/{len(data)}: passed {passed}", flush=True)
    return {"task": "humaneval", "n": len(data), "pass@1": passed / len(data), "not_run_unsafe": skipped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tasks", default="gsm8k,humaneval")
    ap.add_argument("--gsm8k-limit", type=int, default=100)
    ap.add_argument("--humaneval-limit", type=int, default=164)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model, tok, meta = mac_lm.load(args.export, args.device)
    log = open(args.out.replace(".json", "_samples.jsonl"), "w") if args.out else None
    results = []
    for t in args.tasks.split(","):
        t0 = time.time()
        r = gsm8k(model, tok, args.gsm8k_limit, log=log) if t == "gsm8k" else humaneval(model, tok, args.humaneval_limit,
                                                                                       log=log)
        r["seconds"] = round(time.time() - t0)
        results.append(r)
        print(json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"step": meta["step"], "tokens": meta["tokens"], "results": results}, f, indent=1)
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
