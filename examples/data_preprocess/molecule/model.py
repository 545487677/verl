import itertools
import asyncio
from openai import AsyncAzureOpenAI
import itertools
import asyncio
from openai import AsyncAzureOpenAI
import time
from tqdm.asyncio import tqdm
CONCURRENT_REQUEST_LIMIT = 15
import os
from dotenv import load_dotenv
load_dotenv()
import tiktoken

MODEL_CONFIG = {
    "GPT4o-main": {
        "api_base": os.environ.get("GPT4O_MAIN_API_BASE"),
        "api_key": os.environ.get("GPT4O_MAIN_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini": {
        "api_base": os.environ.get("GPT4O_MINI_API_BASE"),
        "api_key": os.environ.get("GPT4O_MINI_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
   "GPT4o-sunquan": {
        "api_base": os.environ.get("GPT4O_SUNQUAN_API_BASE"),
        "api_key": os.environ.get("GPT4O_SUNQUAN_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-sunquan": {
        "api_base": os.environ.get("GPT4O_SUNQUAN_API_BASE"),
        "api_key": os.environ.get("GPT4O_SUNQUAN_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-zhangfei": {
        "api_base": os.environ.get("GPT4O_ZHANGFEI_API_BASE"),
        "api_key": os.environ.get("GPT4O_ZHANGFEI_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-zhangfei": {
        "api_base": os.environ.get("GPT4O_ZHANGFEI_API_BASE"),
        "api_key": os.environ.get("GPT4O_ZHANGFEI_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-athena5": {
        "api_base": os.environ.get("GPT4O_ATHENA5_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA5_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-athena5": {
        "api_base": os.environ.get("GPT4O_ATHENA5_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA5_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-athena6": {
        "api_base": os.environ.get("GPT4O_ATHENA6_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA6_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-athena6": {
        "api_base": os.environ.get("GPT4O_ATHENA6_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA6_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-athena7": {
        "api_base": os.environ.get("GPT4O_ATHENA7_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA7_API_KEY"),
        "model": "gpt-4o",
        "api_version": "2024-03-01-preview"
    },
    "GPT4o-mini-athena7": {
        "api_base": os.environ.get("GPT4O_ATHENA7_API_BASE"),
        "api_key": os.environ.get("GPT4O_ATHENA7_API_KEY"),
        "model": "gpt-4o-mini",
        "api_version": "2024-03-01-preview"
    }

}

class APIModel:
    def __init__(self):
        self.model_pool = self.init_model_pool()
        self.total_tokens = 0  # 初始化总 tokens 数量
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUEST_LIMIT)


    def count_tokens(self, text):
        if not isinstance(text, str):  # 检查是否是字符串
            text = ""  # 如果不是字符串，将其设置为空字符串
        return len(self.tokenizer.encode(text))

    # 初始化模型池
    def init_model_pool(self):
        clients = []
        for config in MODEL_CONFIG.values():
            client = AsyncAzureOpenAI(
                azure_endpoint=config["api_base"],
                api_key=config["api_key"],
                api_version=config["api_version"]
            )
            clients.append((client, config["model"]))
        return itertools.cycle(clients)  # 循环池

    # 异步请求，自动从池中获取模型
    async def request_with_fallback(
        self, 
        text, 
        temperature=0.7, 
        max_retries=3, 
        stream=False
    ):
        attempts = 0
        while attempts < max_retries:
            client, model = next(self.model_pool)  # 从池中获取下一个模型
            # print(f"当前调用的模型: {model}")  # 打印出正在使用的模型名称
            async with self.semaphore:
                try:
                    start_time = time.time()
                    # 执行异步请求
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": text}],
                        temperature=temperature,
                        stream=stream,
                    )
                    elapsed_time = time.time() - start_time
                    # print(f"请求耗时: {elapsed_time:.2f} 秒")  # 打印耗时
                    return response if stream else response.choices[0].message.content
                except Exception as e:
                    print(f"Request failed (attempt {attempts + 1}/{max_retries}): {e}")
                    attempts += 1
                    await asyncio.sleep(2)
        raise Exception("All models failed after maximum retries.")


    # 单条消息请求
    # async def chat(self, text, temperature=1):
    #     response = await self.request_with_fallback(text, temperature=temperature)
    #     return response
    async def chat(self, text, temperature=1):
        tokens = self.count_tokens(text)  # 统计当前输入的 tokens 数量
        self.total_tokens += tokens  # 累计 tokens 数量
        response = await self.request_with_fallback(text, temperature=temperature)
        while not response:  # 如果 response 为空，继续调用直到获取有效值
            response = await self.request_with_fallback(text, temperature=temperature)

        output_tokens = self.count_tokens(response)
        self.total_tokens += output_tokens 
        return response

    # 批量请求
    async def batch_chat(self, text_batch, temperature=1):
        tasks = [self.chat(text, temperature=temperature) for text in text_batch]
        responses = []
        async for response in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            responses.append(await response)
        return responses

# 使用示例
async def main():
    api_model = APIModel()
    prompt = "Please generate the molecular structure (SMILES) based on the following description.\n\n" \
          "### Example:\n" \
          "Description: The molecule is an epoxy(hydroxy)icosatrienoate that is the conjugate base of " \
          "11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid, obtained by deprotonation of the carboxy group; " \
          "major species at pH 7.3. It is a conjugate base of an 11 hydroxy-(14R,15S)-epoxy-(5Z,8Z,12E)-icosatrienoic acid.\n" \
          "Answer: <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>\n\n" \
          "### Now try this:\n" \
          "Description: The molecule is a steroid ester that is methyl (17E)-pregna-4,17-dien-21-oate substituted by " \
          "oxo groups at positions 3 and 11. It is a 3-oxo-Delta(4) steroid, an 11-oxo steroid, a steroid ester and a methyl ester. " \
          "It derives from a hydride of a pregnane.\n" \
          "Let's think step by step and return the final answer in <answer> </answer> tags, " \
          "for example <answer> CCCCC[C@@H]1O[C@@H]1/C=C/C(O)C/C=C\\C/C=C\\CCCC(=O)[O-] </answer>."
    responses = await api_model.batch_chat([prompt] * 5)  # 示例批量请求
    for i, response in enumerate(responses):
        print(f"Response {i+1}: {response}")

# 运行示例
if __name__ == "__main__":
    asyncio.run(main())
