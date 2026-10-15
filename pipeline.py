from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
from openvino import Core
from vector_db import KnowledgeBaseManager
from energy_tracker import HardwareProfiler
from detector import HallucinationDetector

class OpenVINOHallucinationPipeline:
    def __init__(self, model_dir="./openvino_models/qwen_int8", device="CPU"):
        print(f"Loading OpenVINO Model from {model_dir} on device target: {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        core = Core()
        available_devices = core.available_devices
        print(f"Available OpenVINO Hardware Devices: {available_devices}")
        
        target_device = device.upper()
        
        # Validate device availability
        if target_device.startswith("GPU") and not any(d.startswith("GPU") for d in available_devices):
            print("GPU plugin not detected or compatible graphics driver missing. Defaulting to CPU.")
            target_device = "CPU"
        elif target_device == "NPU" and "NPU" not in available_devices:
            print("NPU not found or supported on this system. Defaulting to CPU.")
            target_device = "CPU"
            
        self.model = OVModelForCausalLM.from_pretrained(model_dir, device=target_device)
        
        self.kb = KnowledgeBaseManager()
        self.kb.populate_default_knowledge()
        
        self.profiler = HardwareProfiler()
        self.detector = HallucinationDetector()

    def set_device(self, device_name: str) -> str:
        """Dynamically switches execution device with multi-GPU and NPU safety checks."""
        try:
            core = Core()
            available_devices = core.available_devices
            device_str = device_name.upper()
            
            if device_str.startswith("GPU") and not any(d.startswith("GPU") for d in available_devices):
                self.model.to("cpu")
                return "GPU not supported/found. Fell back to CPU."
                
            if device_str == "NPU" and "NPU" not in available_devices:
                self.model.to("cpu")
                return "NPU not found or supported. Fell back to CPU."
                
            self.model.to(device_str.lower())
            return f"Successfully switched to {device_str}"
        except Exception as e:
            print(f"Device switch fallback to CPU due to: {e}")
            self.model.to("cpu")
            return f"Error targeting {device_name}. Fell back to CPU."

    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        messages = [
            {"role": "system", "content": "You are an expert academic assistant. Answer the question accurately based on standard database textbook concepts."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=0.3)
        
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

    def process_query(self, user_query: str) -> dict:
        self.profiler.start_profiling()
        
        raw_llm_output = self.generate_response(user_query)
        h_score, claim_details = self.detector.evaluate_response(raw_llm_output, self.kb)
        metrics = self.profiler.stop_profiling()
        
        is_flagged = h_score >= 0.30  
        
        return {
            "query": user_query,
            "response": raw_llm_output,
            "hallucination_score_pct": round(h_score * 100, 2),
            "flagged": is_flagged,
            "claims": claim_details,
            "metrics": metrics
        }