import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SmolLMAgent:
    def __init__(self, model_id="HuggingFaceTB/SmolLM2-1.7B-Instruct"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load(self):
        if self.is_loaded: return
        print(f"Loading {self.model_id} on {self.device}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.is_loaded = True
            print("SmolLM loaded successfully.")
        except Exception as e:
            print(f"Error loading SmolLM: {e}")
            
    def generate_response(self, messages, max_new_tokens=256):
        if not self.is_loaded:
            self.load()
            
        if not self.is_loaded:
            return "Sorry, I am currently unavailable. The language model failed to load."
            
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )
        
        response_token_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
        return response

# Global instance
agent = SmolLMAgent()
