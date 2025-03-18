# model_loader.py
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalModelLoader:
    def __init__(self, model_name="deepseek-ai/deepseek-R1-Distill-Llama-8B", quantization="int8"):
        """Initialize the local model loader
        
        Args:
            model_name: Name or path of the model
            quantization: Quantization level (int8, int4, or fp16)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.logger = logging.getLogger("ModelLoader")
        
    def load_model(self):
        """Load the model into GPU memory"""
        if self.loaded:
            return True
            
        try:
            self.logger.info(f"Loading model {self.model_name} with {self.quantization} quantization...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set appropriate device map and quantization
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                
                # Load with appropriate quantization
                if self.quantization == "int8":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        load_in_8bit=True,
                        device_map="auto"
                    )
                elif self.quantization == "int4":
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        load_in_4bit=True,
                        device_map="auto"
                    )
                else:
                    # Half precision
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
            else:
                self.logger.warning("CUDA not available, falling back to CPU (this will be very slow)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
                
            self.loaded = True
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def generate_thought(self, prompt, max_length=200, temperature=0.7, top_p=0.9):
        """Generate a thought using the local model
        
        Args:
            prompt: Text prompt to generate from
            max_length: Maximum number of new tokens
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text
        """
        if not self.loaded and not self.load_model():
            return "Failed to load model"
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True
                )
            
            # Decode only the new tokens, not the prompt
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating thought: {str(e)}")
            return f"Error generating thought: {str(e)}"
    
    def unload_model(self):
        """Unload model from GPU memory to free resources"""
        if not self.loaded:
            return
            
        import gc
        
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        
        self.model = None
        self.loaded = False
        
        self.logger.info("Model unloaded from GPU memory")