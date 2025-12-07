import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HybridCache, Gemma3ForCausalLM, GemmaTokenizerFast, DynamicCache

class CacheObject:
    def __init__(self, length):
        self.history = dict()
        self.length = length
    def add(self, uid):
        if len(self.history) >= self.length:
            oldest_uid = list(self.history.keys())[0]
            del self.history[oldest_uid]
        self.history[uid] = DynamicCache()
    def get(self, uid):
        if uid in self.history:
            return self.history[uid]
        else:
            self.add(uid)
            return self.history[uid]
    def update(self, uid, past_key_values):
        if uid in self.history:
            self.history[uid] = past_key_values
        else:
            self.add(uid)
            self.history[uid] = past_key_values

class Gemma3Object:
    def __init__(self, path = "C:/Users/user/LLM/gemma3", history_len = 32, tools = dict()):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = Gemma3ForCausalLM.from_pretrained(path,
                                        device_map="auto",
                                        quantization_config=quantization_config,
                                        torch_dtype=torch.bfloat16,
                                        low_cpu_mem_usage=True
                                        )
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.msg = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        self.cache = CacheObject(history_len)

    def _prefill(self, input_ids, past_key_values):
        chunks = torch.split(input_ids[:, :-1], 32, dim=-1)
        with torch.no_grad():
            for chunk in chunks:
                self.model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)

    def _decode_step(self, input_ids, past_key_values):
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values)
        return outputs

    def _decode(self, input_ids, past_key_values, stop_words, max_tokens=32768):
        tokenizer = self.tokenizer
        input_ids = input_ids[:, -1:]
        byte_buffer = b''
        
        for _ in range(max_tokens):
            outputs = self._decode_step(input_ids, past_key_values)
            logits = outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            token_id = next_token.item()
            input_ids = next_token

            if token_id in stop_words:
                break

            w = tokenizer.decode(token_id)
            byte_buffer += w.encode('utf-8')
            try:
                word = byte_buffer.decode("utf-8")
                yield word
                byte_buffer = b''
            except:
                pass

    def decision(self, prompt: str):
        model = self.model
        tokenizer = self.tokenizer
        past_key_values = DynamicCache()

        MSG = self.msg.format(prompt=prompt)
        input_ids = torch.tensor(tokenizer.encode(MSG)).to(model.device)
        input_ids = input_ids.unsqueeze(0)
        eos_token_ids = [tokenizer.eos_token_id, 106]

        self._prefill(input_ids, past_key_values)
        input_ids = input_ids[:, -1:]

        outputs = self._decode_step(input_ids, past_key_values)
        logits = outputs.logits
        decide = [9277, 4339][torch.argmax(logits[:, -1, [9277, 4339]], dim=-1, keepdim=False).item()]
        return tokenizer.decode(decide)

    def predict(self, prompt: str, uid: str):
        model = self.model
        tokenizer = self.tokenizer
        past_key_values = self.cache.get(uid)
        
        MSG = self.msg.format(prompt=prompt)
        input_ids = torch.tensor(tokenizer.encode(MSG)).to(model.device)
        input_ids = input_ids.unsqueeze(0)
        eos_token_ids = [tokenizer.eos_token_id, 106]

        self._prefill(input_ids, past_key_values)
        
        try:
            for word in self._decode(input_ids, past_key_values, eos_token_ids):
                yield word
            self.cache.update(uid, past_key_values)
        except KeyboardInterrupt as e:
            pass
        finally:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
