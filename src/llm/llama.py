from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
from transformers import TextStreamer

import logging
import hydra
from omegaconf import DictConfig

from src.telemetry.prometheus import telemetry

logger = logging.getLogger(__name__)

class SciLlama:
    def __init__(self, cfg:DictConfig):
        self.model = None
        self.tokenizer = None
        self.streamer = None
        self.cfg = cfg
        self.model_name = cfg.model.model_name_or_lora_path
        self.max_seq_length = cfg.model.max_seq_length
        self.max_new_tokens = cfg.model.max_new_tokens
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

    def load_model(self):
        logger.info(f'Loading {self.model_name}')
        # Load the saved lora adapter
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Faster inference
        FastLanguageModel.for_inference(self.model)        
        
        # Set chat template for Llama-3.2
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template='llama-3.1'
        )

    @telemetry.track_vram('SciLlama-3.2-3B')
    @telemetry.track_latency(telemetry.llm_latency)
    def generate(self, query, temperature=0.7):
        temperature = max(0.1, min(1.0, temperature))

        prompt = [
            {'role': 'system', 'content': 'You are an expert STEM tutor. Explain STEM subjects clearly, accurately, and concisely for students.'},
            {"role": "user", "content": str(query)},
        ]

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)

        input_len = len(inputs)
        telemetry.llm_tokens.labels('input').inc(input_len)

        self.streamer = TextStreamer(self.tokenizer)
        
        outputs = self.model.generate(
            inputs,
            streamer=self.streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p = 0.9 if temperature > 0.5 else 1.0,
            do_sample= temperature > 0.1,
            pad_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_parts = full_response.split("<|start_header_id|>assistant<|end_header_id|>")
        if len(response_parts) > 1:
            clean_response = response_parts[-1].split("<|eot_id|>")[0].strip()
        else: 
            clean_response = full_response.split("assistant")[-1].strip()
        
        out_len = len(clean_response) - input_len
        telemetry.llm_tokens.labels('output').inc(out_len)

        return clean_response

@hydra.main(config_path='../../conf/llm/', config_name='llama', version_base='1.3')
def test_generation(cfg:DictConfig):
    try:
        llama = SciLlama(cfg)

        llama.generate(query='Explain quantum ML')

    except Exception as e:
        logger.error(f'Generation failed: {e}')
        raise

if __name__ == '__main__':
    test_generation()