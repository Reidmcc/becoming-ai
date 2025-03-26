import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

class LocalModelLoader:
    def __init__(self, model_name="deepseek-ai/deepseek-R1-Distill-Llama-8B", quantization="int8", cache_dir=None):
        """Initialize the local model loader
        
        Args:
            model_name: Name or path of the model
            quantization: Quantization level (int8, int4, or fp16)
            cache_dir: Directory to cache model files (optional)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.logger = logging.getLogger("ModelLoader")
        
    def load_model(self):
        """Load the model into memory"""
        if self.loaded:
            return True
            
        try:
            self.logger.info(f"Loading model {self.model_name} with {self.quantization} quantization...")
            
            # First load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True  # Needed for some models with custom code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set appropriate device map and quantization
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                device_map = "auto"
                
                # Configure quantization
                if self.quantization == "int8":
                    # 8-bit quantization with bitsandbytes
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_enable_fp32_cpu_offload=True
                    )
                elif self.quantization == "int4":
                    # 4-bit quantization with bitsandbytes
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_use_double_quant=True
                    )
                else:
                    # fp16 mode
                    quantization_config = None
                
                # Load the model with appropriate quantization
                if self.quantization in ["int8", "int4"]:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        device_map=device_map,
                        quantization_config=quantization_config,
                        trust_remote_code=True  # Needed for some models with custom code
                    )
                else:
                    # Half precision (fp16)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        cache_dir=self.cache_dir,
                        torch_dtype=torch.float16,
                        device_map=device_map,
                        trust_remote_code=True
                    )
            else:
                self.logger.warning("CUDA not available, falling back to CPU (this will be very slow)")
                # For CPU, avoid quantization as it might not be compatible
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
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
            # Add handling for special tokens needed by some models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs.get("attention_mask", None)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode only the new tokens, not the prompt
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating thought: {str(e)}")
            return f"Error generating thought: {str(e)}"
    
    def unload_model(self):
        """Unload model from memory to free resources"""
        if not self.loaded:
            return
            
        import gc
        
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        self.model = None
        self.loaded = False
        
        self.logger.info("Model unloaded from memory")