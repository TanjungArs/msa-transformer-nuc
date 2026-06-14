import os
import sys
import torch
import argparse
import json
import yaml
from itertools import combinations
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.tokenizer import Tokenizer
from src.model.msa_transformer import MSATransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# MODEL LOADER (AUTO FROM CHECKPOINT)
# ---------------------------------------------------------------------
def load_model_from_checkpoint(ckpt_path, tokenizer, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # --- load model config ---
    if isinstance(ckpt, dict) and "model_cfg" in ckpt:
        model_cfg = ckpt["model_cfg"]
    else:
        # fallback for legacy checkpoint (stage-2)
        with open(os.path.join(BASE_DIR, "configs", "model.yaml")) as f:
            model_cfg = yaml.safe_load(f)["model"]

        # infer max_len from positional embedding
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        for k, v in state.items():
            if k.endswith("pos_embedding.pe"):
                model_cfg["max_len"] = v.size(0)
                break

    # --- build model ---
    model = MSATransformer(
        vocab_size=len(tokenizer.vocab),
        dim=model_cfg["dim"],
        enc_depth=model_cfg["enc_depth"],
        dec_depth=model_cfg["dec_depth"],
        n_heads=model_cfg["n_heads"],
        ff_dim=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
        max_len=model_cfg["max_len"],
        pad_id=tokenizer.pad_id
    ).to(device)

    # --- load weights ---
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    return model, model_cfg

# ---------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------
def infer(model, tokenizer, unaligned, max_len, debug=False):
    input_ids = torch.tensor(
        tokenizer.encode(unaligned, add_eos=False),
        device=device
    ).unsqueeze(0)

    key_padding_mask = (input_ids == tokenizer.pad_id)
    enc = model.encoder(input_ids, key_padding_mask)

    dec = torch.tensor([[tokenizer.sos_id]], device=device)

    for step in range(max_len):
        logits = model.decoder_align(enc, dec, key_padding_mask)

        if debug:
            probs = torch.softmax(logits[:, -1], dim=-1)
            topk = torch.topk(probs, 5)
            print(
                f"[step {step}]",
                [(tokenizer.id_to_token[i.item()], round(p.item(), 3))
                 for i, p in zip(topk.indices[0], topk.values[0])]
            )

        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        dec = torch.cat([dec, next_tok], dim=1)

        if next_tok.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(dec[0].tolist())

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def from_flattened(flat_str, n_seqs):
    seqs = [""] * n_seqs
    for idx, c in enumerate(flat_str):
        seqs[idx % n_seqs] += c
    return seqs

def load_jsonl(file_path):
    unalign_list = []
    with open(file_path) as f:
        for line in f:
            data = json.loads(line)
            unalign_list.append(data["unalign_string"])
    return unalign_list

def sp_score(seqs, match=1, mismatch=-1, gap=-1):
    """Hitung Sum-of-Pairs Score (SP Score) seperti web alignment scorer."""
    sp = 0
    for i, j in combinations(range(len(seqs)), 2):
        s1, s2 = seqs[i], seqs[j]
        for a, b in zip(s1, s2):
            if a == '-' or b == '-':
                sp += gap
            elif a == b:
                sp += match
            else:
                sp += mismatch
    return sp


def evaluate_msa(seqs):
    """Evaluasi MSA dengan logika mirip web alignment scorer, tetap return dict lama."""
    records = [SeqRecord(Seq(s), id=f"seq{i+1}") for i, s in enumerate(seqs)]
    msa = MultipleSeqAlignment(records)

    n_seqs = len(seqs)
    n_cols = msa.get_alignment_length()

    # --- Column-wise calculations ---
    col_score = 0
    total_gaps = 0
    conserved_cols = 0  # untuk identity %
    for col in range(n_cols):
        col_chars = [msa[i, col] for i in range(n_seqs)]
        unique_chars = set(col_chars) - {'-'}
        total_gaps += col_chars.count('-')
        if len(unique_chars) == 1:
            col_score += 1  # fully conserved column
            conserved_cols += 1

    column_score_pct = round(col_score / n_cols * 100, 2)
    gap_pct = round(total_gaps / (n_cols * n_seqs) * 100, 2)
    identity_pct = round(conserved_cols / n_cols * 100, 2)

    # --- Sum-of-Pairs ---
    sp_total = sp_score(seqs, match=1, mismatch=-1, gap=-1)
    max_sp = (n_seqs * (n_seqs - 1) // 2) * n_cols  # semua match
    normalized_sp = round((sp_total + max_sp) / (2 * max_sp) * 100, 2)

    # --- Quality Score ---
    quality_score = round(
        0.4 * identity_pct + 0.3 * column_score_pct + 0.3 * normalized_sp, 2
    )

    return {
        "Identity %": identity_pct,
        "Gap %": gap_pct,
        "Column Score %": column_score_pct,
        "Sum-of-Pairs Score": sp_total,
        "Normalized SP %": normalized_sp,
        "Quality Score": quality_score
    }

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSA Transformer Inference with Evaluation")
    parser.add_argument("--seq", required=True, type=int, help="Number of sequences")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    n_seq = args.seq
    tokenizer = Tokenizer()

    model_path = os.path.join(BASE_DIR, f"experiments/final/stage-{n_seq}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model, model_cfg = load_model_from_checkpoint(
        model_path, tokenizer, device
    )

    print("[INFO] Loaded model config:", model_cfg)

    jsonl_path = f"/home/tyudha/final_dataset/lite/msa-nuc-{n_seq}.jsonl"
    unaligned_list = load_jsonl(jsonl_path)

    output_file = f"inference/msa_infer_seq_{n_seq}.txt"
    os.makedirs("inference", exist_ok=True)

    with open(output_file, "w") as out_f:
        for idx, unalign in enumerate(unaligned_list, 1):
            aligned_flat = infer(
                model,
                tokenizer,
                unalign,
                max_len=model_cfg["max_len"],
                debug=args.debug
            )

            seqs = from_flattened(aligned_flat, n_seq)

            out_f.write(f">Sample {idx}\n")
            for i, s in enumerate(seqs, 1):
                out_f.write(f">seq{i}\n{s}\n")

            eval_dict = evaluate_msa(seqs)
            for k, v in eval_dict.items():
                out_f.write(f"# {k}: {v}\n")
            out_f.write("\n")

    print(f"[OK] Inference done, saved in {output_file}")
