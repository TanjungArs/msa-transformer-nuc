import os
import sys
import torch
import argparse
import json
from itertools import combinations
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ------------------------- Path Setup -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.tokenizer import AlignTokenizer, GapTokenizer
from src.model.msa_transformer import MSATransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------- Inference (ALIGN ONLY) -------------------------
@torch.no_grad()
def infer_align_only(
    model,
    tokenizer: AlignTokenizer,
    unaligned: str,
    max_len: int = 1024,
    debug: bool = False
):
    input_ids = torch.tensor(
        tokenizer.encode(unaligned, add_eos=False),
        device=device
    ).unsqueeze(0)

    key_padding_mask = (input_ids == tokenizer.pad_id)
    encoder_out = model.encoder(input_ids, key_padding_mask)

    decoder_input = torch.tensor([[tokenizer.sos_id]], device=device)

    for step in range(max_len):
        logits = model.decoder_align(
            encoder_out=encoder_out,
            decoder_input=decoder_input,
            enc_padding_mask=key_padding_mask
        )

        if debug:
            probs = torch.softmax(logits[:, -1], dim=-1)
            topk = torch.topk(probs, 5)
            print(
                f"[step {step}]",
                [(tokenizer.id_to_token[i.item()], round(p.item(), 3))
                 for i, p in zip(topk.indices[0], topk.values[0])]
            )

        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        decoder_input = torch.cat([decoder_input, next_tok], dim=1)

        if next_tok.item() == tokenizer.eos_id:
            break

    return tokenizer.decode(decoder_input[0].tolist())


# ------------------------- Unflatten -------------------------
def from_flattened(flat_str: str, n_seqs: int):
    seqs = [""] * n_seqs
    for idx, c in enumerate(flat_str):
        seqs[idx % n_seqs] += c
    return seqs


# ------------------------- Load JSONL -------------------------
def load_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ------------------------- SP Score -------------------------
def sp_score(seqs, match=1, mismatch=-1, gap=-1):
    sp = 0
    for i, j in combinations(range(len(seqs)), 2):
        for a, b in zip(seqs[i], seqs[j]):
            if a == "-" or b == "-":
                sp += gap
            elif a == b:
                sp += match
            else:
                sp += mismatch
    return sp


# ------------------------- Evaluate MSA -------------------------
def evaluate_msa(seqs):
    records = [SeqRecord(Seq(s), id=f"seq{i+1}") for i, s in enumerate(seqs)]
    msa = MultipleSeqAlignment(records)

    n_seqs = len(seqs)
    n_cols = msa.get_alignment_length()

    col_score = 0
    total_gaps = 0

    for col in range(n_cols):
        chars = [msa[i, col] for i in range(n_seqs)]
        if len(set(chars)) == 1:
            col_score += 1
        total_gaps += chars.count("-")

    column_score_pct = round(col_score / n_cols * 100, 2)
    gap_pct = round(total_gaps / (n_cols * n_seqs) * 100, 2)

    sp_total = sp_score(seqs)
    max_sp = (n_seqs * (n_seqs - 1) // 2) * n_cols
    normalized_sp = round((sp_total + max_sp) / (2 * max_sp) * 100, 2)

    quality_score = round(
        0.4 * column_score_pct +
        0.3 * normalized_sp +
        0.3 * (100 - gap_pct),
        2
    )

    return {
        "Column Score %": column_score_pct,
        "Gap %": gap_pct,
        "Sum-of-Pairs Score": sp_total,
        "Normalized SP %": normalized_sp,
        "Quality Score": quality_score
    }


# ------------------------- Main -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MSA Transformer Inference (Alignment Only + Summary)"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    align_tok = AlignTokenizer()
    gap_tok = GapTokenizer()

    # -------- Load Model --------
    model_path = os.path.join(BASE_DIR, "experiments/final/stage-5.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    ckpt = torch.load(model_path, map_location=device)
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
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------- Load Dataset --------
    jsonl_path = os.path.join(BASE_DIR, "data/msa-nuc-2/p.jsonl")
    data_list = load_jsonl(jsonl_path)

    folder_path = os.path.join(BASE_DIR, f"inference/result/msa-nuc-5")
    os.makedirs(folder_path, exist_ok=True)

    out_align = os.path.join(folder_path, "msa_infer_model.txt")
    out_metrics = os.path.join(folder_path, "msa_infer_metrics.txt")
    out_summary = os.path.join(folder_path, "msa_summary.txt")

    # -------- Summary Stats --------
    summary_stats = {}
    metrics_keys = [
        "Column Score %",
        "Gap %",
        "Sum-of-Pairs Score",
        "Normalized SP %",
        "Quality Score"
    ]

    for k in metrics_keys:
        summary_stats[k] = {"Better": 0, "Worse": 0, "Equal": 0}

    summary_stats["Model Avg"] = {k: 0 for k in metrics_keys}
    summary_stats["GT Avg"] = {k: 0 for k in metrics_keys}

    # -------- Inference Loop --------
    with open(out_align, "w") as fa, open(out_metrics, "w") as fm:
        for idx, data in enumerate(data_list, 1):

            unalign = data["unalign_string"]
            gt_align = data.get("aligned_string")
            n_seq = data["seq_count"]

            pred_flat = infer_align_only(
                model,
                align_tok,
                unalign,
                max_len=model_cfg["max_len"],
                debug=args.debug
            )

            seqs_unalign = unalign.split("|")
            seqs_pred = from_flattened(pred_flat, n_seq)
            seqs_gt = from_flattened(gt_align, n_seq) if gt_align else None

            # ---- Save Alignment ----
            fa.write(f">Sample {idx}\n")
            
            for i, s in enumerate(seqs_unalign, 1):
                fa.write(f">seq{i}\n{s}\n")
            fa.write("\n")
            
            for i, s in enumerate(seqs_pred, 1):
                fa.write(f">seq{i}\n{s}\n")
            fa.write("\n")

            # ---- Metrics ----
            eval_pred = evaluate_msa(seqs_pred)
            eval_gt = evaluate_msa(seqs_gt) if seqs_gt else None

            fm.write(f">Sample {idx}\n")

            for k in metrics_keys:
                model_v = eval_pred[k]
                gt_v = eval_gt[k] if eval_gt else "N/A"

                cmp = ""
                if gt_v != "N/A":
                    if model_v == gt_v:
                        cmp = "Equal"
                    elif k == "Gap %":
                        cmp = "Better" if model_v < gt_v else "Worse"
                    else:
                        cmp = "Better" if model_v > gt_v else "Worse"

                    summary_stats[k][cmp] += 1
                    summary_stats["Model Avg"][k] += model_v
                    summary_stats["GT Avg"][k] += gt_v

                fm.write(f"# {k}: Model={model_v} | GT={gt_v} | {cmp}\n")

            fm.write("\n")

    # -------- Write Summary --------
    total_samples = len(data_list)

    with open(out_summary, "w") as fs:
        fs.write("[Summary]\n")
        fs.write(f"Total samples tested: {total_samples}\n\n")

        for k in metrics_keys:
            better = summary_stats[k]["Better"]
            equal = summary_stats[k]["Equal"]
            worse = summary_stats[k]["Worse"]

            avg_model = round(summary_stats["Model Avg"][k] / total_samples, 2)
            avg_gt = round(summary_stats["GT Avg"][k] / total_samples, 2)

            fs.write(
                f"{k}: "
                f"Better={better} | "
                f"Equal={equal} | "
                f"Worse={worse} | "
                f"Avg Model={avg_model} | "
                f"Avg GT={avg_gt}\n"
            )

    print(f"[OK] Alignment saved to {out_align}")
    print(f"[OK] Metrics saved to {out_metrics}")
    print(f"[OK] Summary saved to {out_summary}")
