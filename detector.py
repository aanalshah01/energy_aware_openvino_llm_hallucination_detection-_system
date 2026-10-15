import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class HallucinationDetector:
    def __init__(self, nli_model_name="roberta-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)

    def extract_atomic_claims(self, text: str) -> list[str]:
        """Decomposes LLM text into individual verifiable sentences/claims."""
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 8]
        return claims

    def verify_claim_against_evidence(self, claim: str, evidence_list: list[str]) -> dict:
        """Runs NLI checks (Entailment / Neutral / Contradiction) across evidence."""
        if not evidence_list:
            return {"status": "UNSUPPORTED", "nli_label": "NO_EVIDENCE", "confidence": 0.0}

        best_status = "UNSUPPORTED"
        best_label = "CONTRADICTION"
        max_entailment_prob = 0.0

        for evidence in evidence_list:
            inputs = self.tokenizer(evidence, claim, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
            
            # RoBERTa-MNLI labels: 0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT
            entail_prob = probs[2].item()
            contra_prob = probs[0].item()

            if entail_prob > max_entailment_prob:
                max_entailment_prob = entail_prob
                if entail_prob >= 0.5:
                    best_status = "SUPPORTED"
                    best_label = "ENTAILMENT"
                elif contra_prob >= 0.5:
                    best_status = "CONTRADICTION"
                    best_label = "CONTRADICTION"
                else:
                    best_status = "UNSUPPORTED"
                    best_label = "NEUTRAL"

        return {
            "status": best_status,
            "nli_label": best_label,
            "confidence": round(max_entailment_prob, 3)
        }

    def evaluate_response(self, text: str, kb_manager) -> tuple[float, list[dict]]:
        claims = self.extract_atomic_claims(text)
        if not claims:
            return 0.0, []

        verified_claims = []
        supported_count = 0

        for claim in claims:
            evidences = kb_manager.search_evidence(claim, n_results=2)
            result = self.verify_claim_against_evidence(claim, evidences)
            
            if result["status"] == "SUPPORTED":
                supported_count += 1
                
            verified_claims.append({
                "claim": claim,
                "evidence": evidences[0] if evidences else "No matching evidence found in KB.",
                "status": result["status"],
                "label": result["nli_label"],
                "confidence": result["confidence"]
            })

        # Score = 1 - (Supported Claims / Total Claims)
        hallucination_score = 1.0 - (supported_count / len(claims))
        return round(hallucination_score, 4), verified_claims