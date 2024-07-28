import torch
import transformers

model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer.save_pretrained('models/meta-llama-tokenizer')
model.save_pretrained('models/meta-llama-model')
