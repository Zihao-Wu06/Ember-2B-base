#!/usr/bin/env python
"""MBPP (Austin et al. 2021) on an exported checkpoint on the Mac, greedy pass@1.

Standard setting (the original paper; lm-eval-harness `mbpp`): the 500 test problems (task_id 11-510), 3-shot with the
prompt-split problems 2, 3 and 4. Each shot is

    You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:

    {assert 1}
    {assert 2}
    {assert 3}
    [BEGIN]
    {reference code}
    [DONE]

the shots are separated by blank lines, and the test problem ends at "[BEGIN]\\n". Generation stops at [DONE]. The
completion runs after test_setup_code and before the three asserts in a separate `python -I` process
(gen_eval.run_program: temp dir, 10 s limit). The few-shot block is encoded once and its decode cache reused, as in
gen_eval.gsm8k. Same file-system / process filter as HumanEval, except that plain `sys` is allowed (two reference
solutions use sys.maxsize).

    .venv-mac/bin/python runs/mac_eval/mbpp_eval.py --export runs/mac_eval/ckpt_0045000 \\
        --out runs/mac_eval/mbpp_0045000.json"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mac_lm  # noqa: E402
import decode  # noqa: E402
import gen_eval  # noqa: E402

assert "(os|sys|" in gen_eval.UNSAFE.pattern
UNSAFE = re.compile(gen_eval.UNSAFE.pattern.replace("(os|sys|", "(os|"))
STOPS = ("[DONE]", "\nYou are an expert", "[BEGIN]")
SHOT_IDS = (2, 3, 4)


def block(d, code=None):
    s = (f"You are an expert Python programmer, and here is your task: {d['text']} Your code should pass these tests:"
         "\n\n" + "\n".join(d["test_list"]) + "\n[BEGIN]\n")
    if code is not None:
        s += code.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n[DONE]"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model, tok, meta = mac_lm.load(args.export, args.device)
    prompt = {d["task_id"]: d for d in gen_eval.read_jsonl("mbpp", "full", "prompt-00000-of-00001.jsonl")}
    test = gen_eval.read_jsonl("mbpp", "full", "test-00000-of-00001.jsonl")[args.offset:args.offset + args.limit]
    prefix = "\n\n".join(block(prompt[i], prompt[i]["code"]) for i in SHOT_IDS) + "\n\n"
    cache = decode.prefill(model, mac_lm.encode(tok, prefix))
    log = open(args.out.replace(".json", "_samples.jsonl"), "w") if args.out else None
    passed = skipped = no_done = 0
    t_start = time.time()

    def stop(out):
        return any(s in tok.decode(out) for s in STOPS)

    for i, d in enumerate(test):
        t0 = time.time()
        ids = decode.generate_ids(model, tok.encode(block(d)).ids, args.max_new, stop, init=cache)
        text = tok.decode(ids)
        no_done += "[DONE]" not in text
        comp = text[:min([text.find(s) for s in STOPS if s in text] or [len(text)])].rstrip()
        if UNSAFE.search(comp):
            ok, err = False, "not run (unsafe pattern)"
            skipped += 1
        else:
            ok, err = gen_eval.run_program(d["test_setup_code"] + "\n" + comp + "\n\n" + "\n".join(d["test_list"])
                                           + "\n")
        passed += ok
        if log is not None:
            log.write(json.dumps({"task_id": d["task_id"], "passed": ok, "completion": comp,
                                  "error": "" if ok else err, "seconds": round(time.time() - t0, 1)},
                                 ensure_ascii=False) + "\n")
            log.flush()
        if (i + 1) % 10 == 0 or i + 1 == len(test):
            el = time.time() - t_start
            print(f"  mbpp {i + 1}/{len(test)}: passed {passed} ({100 * passed / (i + 1):.1f}%), "
                  f"{el / (i + 1):.1f} s/problem, eta {el / (i + 1) * (len(test) - i - 1) / 60:.0f} min", flush=True)
    res = {"step": meta["step"], "tokens": meta["tokens"], "task": "mbpp", "shots": list(SHOT_IDS),
           "offset": args.offset, "n": len(test), "passed": passed, "pass@1": round(100 * passed / len(test), 2),
           "not_run_unsafe": skipped, "no_done": no_done, "seconds": round(time.time() - t_start)}
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
    print("MBPP_DONE", flush=True)


if __name__ == "__main__":
    main()
