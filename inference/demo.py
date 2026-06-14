import os
import sys
import torch
import argparse
import subprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.tokenizer import AlignTokenizer, GapTokenizer
from src.model.msa_transformer import MSATransformer
from pymsa.core.msa import MSA
from pymsa.core.score import SumOfPairs, PercentageOfNonGaps

# =========================
# ARGUMENT
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--fasta", type=str, required=True)
parser.add_argument("--outdir", type=str, default="output_single")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(args.outdir, exist_ok=True)

# =========================
# FASTA
# =========================
def read_fasta(file_path):
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

def count_seq(file_path):
    with open(file_path) as f:
        return sum(1 for line in f if line.startswith(">"))

def to_fasta(seqs):
    lines = []
    for i, s in enumerate(seqs, 1):
        lines.append(f">seq{i}")
        lines.append(s)
    return "\n".join(lines)

# =========================
# FORMAT
# =========================
def to_concat(seqs):
    return "|".join(seqs)

def from_flattened(flat_str, n_seqs):
    assert len(flat_str) % n_seqs == 0, "Output model tidak valid!"
    seqs = [""] * n_seqs
    for i, c in enumerate(flat_str):
        seqs[i % n_seqs] += c
    return seqs

# =========================
# METRICS
# =========================
def compute_metrics(seqs):
    seqs = [s.upper() for s in seqs]
    max_len = max(len(s) for s in seqs)
    seqs = [s.ljust(max_len, '-') for s in seqs]

    msa = MSA(seqs)

    sp = SumOfPairs(msa).compute()
    cs = PercentageOfNonGaps(msa).compute()

    gaps = sum(s.count("-") for s in seqs)

    return round(sp, 4), round(cs, 4), gaps

# =========================
# EXTERNAL TOOLS
# =========================
def run(cmd, name):
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] {name} gagal")
        return False
    return True

def run_mafft(inp, out):
    return run(f"mafft --auto {inp} > {out}", "MAFFT")

def run_clustalo(inp, out):
    return run(f"clustalo -i {inp} -o {out} --force", "CLUSTALO")

def run_muscle(inp, out):
    return run(f"muscle -align {inp} -output {out}", "MUSCLE")

# =========================
# MODEL INFERENCE
# =========================
@torch.no_grad()
def infer(model, tokenizer, unalign, device, max_len=2048):
    input_ids = torch.tensor(
        tokenizer.encode(unalign, add_eos=False),
        device=device
    ).unsqueeze(0)

    key_padding_mask = (input_ids == tokenizer.pad_id)
    encoder_out = model.encoder(input_ids, key_padding_mask)

    decoder_input = torch.tensor([[tokenizer.sos_id]], device=device)

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

# =========================
# MAIN
# =========================
print("[INFO] Membaca FASTA...")
seqs = read_fasta(args.fasta)
n_seqs = count_seq(args.fasta)

if n_seqs < 2:
    raise ValueError("Minimal 2 sequence")

print(f"[INFO] Jumlah sequence: {n_seqs}")

# =========================
# AUTO LOAD MODEL
# =========================
model_path = os.path.join(BASE_DIR, f"experiments/final/stage-{n_seqs}.pt")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

print(f"[INFO] Load model: {model_path}")

align_tok = AlignTokenizer()
gap_tok = GapTokenizer()

ckpt = torch.load(model_path, map_location=DEVICE)
cfg = ckpt["model_cfg"]

model = MSATransformer(
    align_vocab_size=len(align_tok.vocab),
    gap_vocab_size=len(gap_tok.vocab),
    dim=cfg["dim"],
    enc_depth=cfg["enc_depth"],
    dec_depth=cfg["dec_depth"],
    n_heads=cfg["n_heads"],
    ff_dim=cfg["ff_dim"],
    dropout=cfg["dropout"],
    max_len=cfg["max_len"],
    align_pad_id=align_tok.pad_id,
    gap_pad_id=gap_tok.pad_id
).to(DEVICE)

model.load_state_dict(ckpt["model"])
model.eval()

# =========================
# INFER MODEL
# =========================
print("[INFO] Infer model...")
unalign = to_concat(seqs)
pred_flat = infer(model, align_tok, unalign, DEVICE)
model_seqs = from_flattened(pred_flat, n_seqs)

# =========================
# RUN TOOLS
# =========================
temp_in = os.path.join(args.outdir, "input.fasta")
with open(temp_in, "w") as f:
    f.write(to_fasta(seqs))

paths = {
    "mafft": os.path.join(args.outdir, "mafft.fasta"),
    "clustalo": os.path.join(args.outdir, "clustalo.fasta"),
    "muscle": os.path.join(args.outdir, "muscle.fasta"),
}

print("[INFO] Running MAFFT...")
run_mafft(temp_in, paths["mafft"])

print("[INFO] Running CLUSTALO...")
run_clustalo(temp_in, paths["clustalo"])

print("[INFO] Running MUSCLE...")
run_muscle(temp_in, paths["muscle"])

def read_out(p):
    return read_fasta(p)

mafft_seqs = read_out(paths["mafft"])
clustalo_seqs = read_out(paths["clustalo"])
muscle_seqs = read_out(paths["muscle"])

# =========================
# SAVE ALIGNMENTS
# =========================
align_file = os.path.join(args.outdir, "all_alignments.txt")

with open(align_file, "w") as f:
    f.write("===== MODEL =====\n")
    f.write(to_fasta(model_seqs) + "\n\n")

    f.write("===== MAFFT =====\n")
    f.write(to_fasta(mafft_seqs) + "\n\n")

    f.write("===== CLUSTALO =====\n")
    f.write(to_fasta(clustalo_seqs) + "\n\n")

    f.write("===== MUSCLE =====\n")
    f.write(to_fasta(muscle_seqs) + "\n")

# =========================
# SAVE METRICS
# =========================
metrics_file = os.path.join(args.outdir, "metrics.txt")

with open(metrics_file, "w") as f:
    header = f"{'Method':<10}{'SP':>10}{'CS':>10}{'Gaps':>10}\n"
    f.write(header)

    for name, s in {
        "MODEL": model_seqs,
        "MAFFT": mafft_seqs,
        "CLUSTALO": clustalo_seqs,
        "MUSCLE": muscle_seqs,
    }.items():
        sp, cs, gaps = compute_metrics(s)
        f.write(f"{name:<10}{sp:>10}{cs:>10}{gaps:>10}\n")
        os.system('cls||clear')

        print("\033[1mRESULTS\033[0m")
        print(f"Jumlah sequence: {n_seqs}")
        
        print("=" * 44)
        print(f"{'Method':<10} | {'SP':>8} | {'CS':>8} | {'Gaps':>8}")
        print("-" * 44)

        results = {
            "MODEL": model_seqs,
            "MAFFT": mafft_seqs,
            "CLUSTALO": clustalo_seqs,
            "MUSCLE": muscle_seqs,
        }

        for name, s in results.items():
            sp, cs, gaps = compute_metrics(s)
            print(f"{name:<10} | {sp:>8.1f} | {cs:>8.1f} | {gaps:>8}")

        print("=" * 44)
        
# =========================
# CLEANUP
# =========================
os.remove(temp_in)

for p in paths.values():
    if os.path.exists(p):
        os.remove(p)

print(f"Hasil alignment : {align_file}")
print(f"Hasil metrics   : {metrics_file}")