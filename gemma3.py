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
    def __init__(self, path = "C:/Users/user/LLM/gemma3", history_len = 32):
        # ----- Load Qwen Model ----- #
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

        # ----- Chat Template ----- #
        self.msg = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

        # ----- cache dictionary ----- #
        self.cache = CacheObject(history_len)
        
    def predict(self, prompt: str, uid: str):
        # ----- Model & Tokenizer & past_key_values ----- #
        model = self.model
        tokenizer = self.tokenizer
        past_key_values = self.cache.get(uid)
        
        # ----- 結果儲存 ----- #
        res = list()

        # ----- Prompt token產生 ----- #
        MSG = self.msg.format(prompt=prompt)
        input_ids = torch.tensor(tokenizer.encode(MSG)).to(model.device)
        input_ids = input_ids.unsqueeze(0)
        eos_token_ids = [tokenizer.eos_token_id, 106]

        # ----- Prefill ----- #
        chunks = torch.split(input_ids[:, :-1], 32, dim=-1)
        st = 0
        ed = 0
        with torch.no_grad():
            for chunk in chunks:
                ed = st + chunk.shape[1]
                outputs = model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)
                st = ed
        
        # ----- Auto Regressive生成 ----- #
        input_ids = input_ids[:, -1:]
        try:
            byte_buffer = b''
            for _ in range(32768):
                with torch.no_grad():
                    # ----- Update position ----- #
                    ed += 1

                    # ----- Update model kwargs ----- #
                    cache_position = torch.arange(ed-1, ed, dtype=torch.long, device = model.device)

                    # ----- 生成token ----- #
                    outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, cache_position=cache_position)
                    logits = outputs.logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    token_id = next_token.item()
                    input_ids = next_token

                    # ----- 判斷是否終止 ----- #
                    if token_id in eos_token_ids:
                        break

                    # ----- 紀錄token ----- #
                    res += [tokenizer.decode(token_id)]
                    
                    # ----- 輸出文字字串 ----- #
                    byte_buffer += res[-1].encode('utf-8')
                    try:
                        word = byte_buffer.decode("utf-8")
                        yield word
                        byte_buffer = b''
                    except:
                        pass
            self.cache.update(uid, past_key_values)
        except KeyboardInterrupt as e:
            pass
        finally:
            for item in ("input_ids", "outputs", "ogits", "next_token", "token_id"):
                try:
                    eval(f"del {item}")
                except:
                    pass
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
