class AlignTokenizer:
    def __init__(self):
        self.vocab = [
            "<PAD>", "<SOS>", "<EOS>",
            "A", "T", "G", "C", "|", "-"
        ]
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        
        self.pad_id = self.token_to_id["<PAD>"]
        self.sos_id = self.token_to_id["<SOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        
    def encode(self, text: str, add_eos: bool = True):
        align_ids = [self.token_to_id[c] for c in text]
        if add_eos:
            align_ids.append(self.eos_id)
        return align_ids
    
    def decode(self, align_ids: list[int], skip_special: bool = True):
        out = []
        for i in align_ids:
            tok = self.id_to_token[int(i)]
            if skip_special and tok in {"<PAD>", "<SOS>", "<EOS>"}:
                continue
            out.append(tok)
        return "".join(out)
    
class GapTokenizer:
    def __init__(self):
        self.vocab = [
            "<PAD>", "<SOS>", "<EOS>",
            "#", "~"
        ]
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}
        
        self.pad_id = self.token_to_id["<PAD>"]
        self.sos_id = self.token_to_id["<SOS>"]
        self.eos_id = self.token_to_id["<EOS>"]
        
    def encode(self, text: str, add_eos: bool = True):
        gap_ids = [self.token_to_id[c] for c in text]
        if add_eos:
            gap_ids.append(self.eos_id)
        return gap_ids
    
    def decode(self, gap_ids: list[int], skip_special: bool = True):
        out = []
        for i in gap_ids:
            tok = self.id_to_token[int(i)]
            if skip_special and tok in {"<PAD>", "<SOS>", "<EOS>"}:
                continue
            out.append(tok)
        return "".join(out)
