import tqdm
import torch
import logging
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HybridCache, Gemma3ForCausalLM, GemmaTokenizerFast, DynamicCache

class Gemma3Object:
    def __init__(self, path = "D://LLM//gemma//gemma3_4b", history_len = 32):
        # ----- Load Qwen Model ----- #
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = Gemma3ForCausalLM.from_pretrained(path,
                                        device_map="cuda",
                                        quantization_config=quantization_config,
                                        torch_dtype=torch.bfloat16,
                                        )
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path)

        # ----- Chat Template ----- #
        self.msg = "<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"

    def decision(self, prompt: str):
        model = self.model
        tokenizer = self.tokenizer
        past_key_values = DynamicCache()

        # ----- Prompt token產生 ----- #
        MSG = self.msg.format(prompt="[針對所有問題，如果回答的資訊是必須為最新資訊，需要使用外部工具來檢索才能回答，則請回傳True，如果只是一般的問答不需要外部工具請回傳False]\n"+prompt)
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
        input_ids = input_ids[:, -1:]

        # ----- 取得True或False的Token機率 ----- #
        """False, True : 4339, 9277"""
        with torch.no_grad():
            ed += 1
            cache_position = torch.arange(ed-1, ed, dtype=torch.long, device = model.device)
            outputs = model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, cache_position=cache_position)
            logits = outputs.logits
        decide = [9277, 4339][torch.argmax(logits[:, -1, [9277, 4339]], dim=-1, keepdim=False).item()]
        return tokenizer.decode(decide)
    
    def function_call(self, prompt: str, tools: list):
        """
        tools: [{"name": "工具名稱", "description": "工具描述", "parameters": "參數敘述"}, ...]
        """
        # ----- 將tools資訊整合到prompt中 ----- #
        tool_descriptions = ""
        for tool in tools:
            tool_descriptions += f'工具名稱: {tool["name"]}\n工具描述: {tool["description"]}\n參數敘述: {tool["parameters"]}\n\n'

        full_prompt = f"以下是可用的工具：\n{tool_descriptions}\n"
        full_prompt += "目前問題是{prompt}，請根據的需求選擇最合適的工具並生成相應的函數調用。\n請以JSON格式返回函數調用，包含工具名稱和參數。"
        return self.chat(full_prompt, uid="function_call")

    def chat(self, prompt: str, uid: str, force: str = ""):
        # ----- Model & Tokenizer & past_key_values ----- #
        print(f"definition")
        past_key_values = DynamicCache()
        
        # ----- 結果儲存 ----- #
        res = list()

        # ----- Prompt token產生 ----- #
        print(f"prompt")
        MSG = self.msg.format(prompt=prompt)+force
        input_ids = torch.tensor(self.tokenizer.encode(MSG)).to(self.model.device)
        input_ids = input_ids.unsqueeze(0)
        eos_token_ids = [self.tokenizer.eos_token_id, 106]

        # ----- Prefill ----- #
        print(f"prefilling")
        chunks = torch.split(input_ids[:, :-1], 128, dim=-1)
        st = 0
        ed = 0
        with tqdm.tqdm(total=len(chunks), desc="Prefill Tokens") as pbar:
            with torch.no_grad():
                for chunk in chunks:
                    ed = st + chunk.shape[1]
                    outputs = self.model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)
                    st = ed
                    pbar.update()
        
        # ----- Auto Regressive生成 ----- #
        print(f"decoding")
        input_ids = input_ids[:, -1:]
        try:
            byte_buffer = b''
            for _ in range(32768):
                with torch.no_grad():
                    # ----- Update position ----- #
                    ed += 1

                    # ----- Update model kwargs ----- #
                    cache_position = torch.arange(ed-1, ed, dtype=torch.long, device = self.model.device)

                    # ----- 生成token ----- #
                    outputs = self.model(input_ids=input_ids, use_cache=True, past_key_values=past_key_values, cache_position=cache_position)
                    logits = outputs.logits
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    token_id = next_token.item()
                    input_ids = next_token

                    # ----- 判斷是否終止 ----- #
                    if token_id in eos_token_ids:
                        break

                    # ----- 紀錄token ----- #
                    res += [self.tokenizer.decode(token_id)]
                    
                    # ----- 輸出文字字串 ----- #
                    byte_buffer += res[-1].encode('utf-8')
                    try:
                        word = byte_buffer.decode("utf-8")
                        yield word
                        byte_buffer = b''
                    except:
                        pass
        except KeyboardInterrupt as e:
            pass
        finally:
            for item in ("input_ids", "outputs", "ogits", "next_token", "token_id", "past_key_values"):
                try:
                    eval(f"del {item}")
                except:
                    pass
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
