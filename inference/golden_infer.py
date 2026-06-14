import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import torch
import argparse
import json
import subprocess

from src.tokenizer import AlignTokenizer, GapTokenizer
from src.model.msa_transformer import MSATransformer
from pymsa.core.msa import MSA
from pymsa.core.score import SumOfPairs, PercentageOfNonGaps

parser = argparse.ArgumentParser()
parser.add_argument("--seq", type=int, required=True)
args = parser.parse_args()
seq_count = args.seq

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = os.path.join(BASE_DIR, f"data/msa-nuc-{seq_count}/test.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, f"inference/result/msa-nuc-{seq_count}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = {
    "unalign": os.path.join(OUTPUT_DIR, "unalign.txt"),
    "model": os.path.join(OUTPUT_DIR, "aligned_model.txt"),
    "sparta": os.path.join(OUTPUT_DIR, "aligned_sparta.txt"),
    "mafft": os.path.join(OUTPUT_DIR, "aligned_mafft.txt"),
    "clustalo": os.path.join(OUTPUT_DIR, "aligned_clustalo.txt"),
    "muscle": os.path.join(OUTPUT_DIR, "aligned_muscle.txt"),
}

SUMMARY_FILE = os.path.join(OUTPUT_DIR, "metrics_summary.txt")

def load_jsonl(file_path):
    with open(file_path) as f:
        return [json.loads(line) for line in f]

def from_flattened(flat_str, n_seqs):
    seqs = [""] * n_seqs
    for idx, c in enumerate(flat_str):
        seqs[idx % n_seqs] += c
    return seqs

def from_concat(concat_str):
    return concat_str.split("|")

def to_fasta(seqs):
    lines = []
    for i, s in enumerate(seqs, 1):
        lines.append(f">seq{i}")
        lines.append(s)
    return "\n".join(lines)

def append_to_file(file_path, content):
    with open(file_path, "a") as f:
        f.write(content + "\n")

def write_unaligned_sample(file_path, sample_id, seqs):
    append_to_file(file_path, f"Sample {sample_id}")
    append_to_file(file_path, to_fasta(seqs))
    append_to_file(file_path, "")

def write_alignment_sample(file_path, sample_id, seqs, sp, cs, gaps, gap_pct):
    append_to_file(file_path, f"Sample {sample_id}")
    append_to_file(file_path, to_fasta(seqs))
    append_to_file(
        file_path,
        f"\nSP: {round(sp,4)} | CS: {round(cs,2)} | Gaps: {gaps} | Gap%: {gap_pct}"
    )
    append_to_file(file_path, "="*50 + "\n")

def read_fasta_seqs(file_path):
    seqs = []
    current = ""
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    seqs.append(current.upper())
                current = ""
            else:
                current += line
        if current:
            seqs.append(current.upper())
    return seqs

def run_mafft(input_fasta, output_fasta):
    subprocess.run(f"mafft --auto {input_fasta} > {output_fasta}", shell=True, check=True)

def run_clustalo(input_fasta, output_fasta):
    subprocess.run(f"clustalo -i {input_fasta} -o {output_fasta} --force", shell=True, check=True)

def run_muscle(input_fasta, output_fasta):
    subprocess.run(f"muscle -align {input_fasta} -output {output_fasta}", shell=True, check=True)

def compute_metrics(seqs):
    seqs = [s.upper() for s in seqs]
    max_len = max(len(s) for s in seqs)
    seqs = [s.ljust(max_len, '-') for s in seqs]
    msa = MSA(seqs)
    sp = SumOfPairs(msa).compute()
    cs = PercentageOfNonGaps(msa).compute()
    total_gaps = sum(s.count("-") for s in seqs)
    gap_pct = total_gaps / (len(seqs) * max_len) * 100
    return sp, cs, total_gaps, round(gap_pct, 2)

@torch.no_grad()
def infer_align_only(model, tokenizer, unaligned, max_len=1024):
    input_ids = torch.tensor(tokenizer.encode(unaligned, add_eos=False), device=DEVICE).unsqueeze(0)
    key_padding_mask = (input_ids == tokenizer.pad_id)
    encoder_out = model.encoder(input_ids, key_padding_mask)
    decoder_input = torch.tensor([[tokenizer.sos_id]], device=DEVICE)

    for _ in range(max_len):
        logits = model.decoder_align(
            encoder_out=encoder_out,
            decoder_input=decoder_input,
            enc_padding_mask=key_padding_mask
        )
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        decoder_input = torch.cat([decoder_input, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(decoder_input[0].tolist())

def main():
    align_tok = AlignTokenizer()
    gap_tok = GapTokenizer()

    model_path = os.path.join(BASE_DIR, f"experiments/final/stage-{seq_count}.pt")
    ckpt = torch.load(model_path, map_location=DEVICE)
    model_cfg = ckpt["model_cfg"]

    model = MSATransformer(
        align_vocab_size=len(align_tok.vocab),
        gap_vocab_size=len(gap_tok.vocab),
        dim=model_cfg["dim"],
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        n_heads=model_cfg["n_heads"],
        ff_dim=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
        max_len=model_cfg["max_len"],
        align_pad_id=align_tok.pad_id,
        gap_pad_id=gap_tok.pad_id
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"])
    model.eval()

    data_list = load_jsonl(DATA_PATH)

    for f in FILES.values():
        open(f, "w").close()

    summary = {m: {"SP": [], "CS": [], "Gaps": [], "Gap%": []}
               for m in ["model", "sparta", "mafft", "clustalo", "muscle"]}

    for idx, data in enumerate(data_list, 1):
        seqs_unalign = from_concat(data["unalign_string"])
        write_unaligned_sample(FILES["unalign"], idx, seqs_unalign)

    for idx, data in enumerate(data_list, 1):
        pred_flat = infer_align_only(model, align_tok, data["unalign_string"])
        seqs_pred = from_flattened(pred_flat, data["seq_count"])
        sp, cs, gaps, gap_pct = compute_metrics(seqs_pred)
        write_alignment_sample(FILES["model"], idx, seqs_pred, sp, cs, gaps, gap_pct)
        summary["model"]["SP"].append(sp)
        summary["model"]["CS"].append(cs)
        summary["model"]["Gaps"].append(gaps)
        summary["model"]["Gap%"].append(gap_pct)

    for idx, data in enumerate(data_list, 1):
        seqs_sparta = from_flattened(data["aligned_string"], data["seq_count"])
        sp, cs, gaps, gap_pct = compute_metrics(seqs_sparta)
        write_alignment_sample(FILES["sparta"], idx, seqs_sparta, sp, cs, gaps, gap_pct)
        summary["sparta"]["SP"].append(sp)
        summary["sparta"]["CS"].append(cs)
        summary["sparta"]["Gaps"].append(gaps)
        summary["sparta"]["Gap%"].append(gap_pct)

    for method, runner in [("mafft", run_mafft),
                           ("clustalo", run_clustalo),
                           ("muscle", run_muscle)]:

        temp_fasta = os.path.join(OUTPUT_DIR, f"temp_{method}.fasta")
        temp_out = os.path.join(OUTPUT_DIR, f"temp_{method}_out.fasta")

        for idx, data in enumerate(data_list, 1):
            seqs_unalign = from_concat(data["unalign_string"])

            with open(temp_fasta, "w") as f:
                f.write(to_fasta(seqs_unalign))

            runner(temp_fasta, temp_out)

            ext_seqs = read_fasta_seqs(temp_out)
            sp, cs, gaps, gap_pct = compute_metrics(ext_seqs)

            write_alignment_sample(FILES[method], idx, ext_seqs, sp, cs, gaps, gap_pct)

            summary[method]["SP"].append(sp)
            summary[method]["CS"].append(cs)
            summary[method]["Gaps"].append(gaps)
            summary[method]["Gap%"].append(gap_pct)

        if os.path.exists(temp_fasta):
            os.remove(temp_fasta)
        if os.path.exists(temp_out):
            os.remove(temp_out)

    header = f"{'Method':<10}{'Avg SP':>10}{'Avg CS':>10}{'Avg Gaps':>12}{'Avg Gap%':>12}\n"
    with open(SUMMARY_FILE, "w") as f:
        f.write(header)
        for method in summary:
            sp_avg = round(sum(summary[method]["SP"]) / len(summary[method]["SP"]), 2)
            cs_avg = round(sum(summary[method]["CS"]) / len(summary[method]["CS"]), 2)
            gaps_avg = round(sum(summary[method]["Gaps"]) / len(summary[method]["Gaps"]), 2)
            gap_pct_avg = round(sum(summary[method]["Gap%"]) / len(summary[method]["Gap%"]), 2)
            f.write(f"{method:<10}{sp_avg:>10}{cs_avg:>10}{gaps_avg:>12}{gap_pct_avg:>12}\n")

    print("[OK] Semua alignment dan summary selesai.")

if __name__ == "__main__":
    main()