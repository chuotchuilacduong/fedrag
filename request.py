import os
import time
import uuid
import requests 
import aiohttp # [ADD] Thêm thư viện aiohttp để xử lý async
import asyncio
from dotenv import load_dotenv

# Import thư viện Local Embedding
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()

class TimelyClient:
    def __init__(self, api_key=None, base_url=None, model_name="gpt-4o-mini"):
        self.api_key = api_key or os.getenv("TIMELY_API_KEY")
        self.base_url = base_url or os.getenv("TIMELY_BASE_URL", "https://hello.timelygpt.co.kr/api/v2/chat")
        
        if not self.api_key:
            raise ValueError(" Chưa thiết lập TIMELY_API_KEY!")

        self.model_name = model_name
        self.session_id = f"python_agent_{uuid.uuid4()}"
        
        self.access_token = None
        self.token_expires_at = 0

        if SentenceTransformer:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.embedder = None

    def _ensure_auth(self):

        if self.access_token and time.time() < self.token_expires_at:
            return

        url = f"{self.base_url}/sdk-auth/authenticate"
        headers = {
            "Content-Type": "application/json",
            "X-Timely-API": self.api_key
        }

        try:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            
            data = resp.json()
            if not data.get("success") or not data.get("data", {}).get("access_token"):
                raise Exception(f"Auth failed: {data}")

            self.access_token = data["data"]["access_token"]
            self.token_expires_at = time.time() + (55 * 60) 
            print("✅ [AUTH] Đã lấy Token thành công!")
            
        except Exception as e:
            print(f" [AUTH] Lỗi xác thực: {e}")
            raise

    def _process_response(self, data):
        msg_type = data.get("type")
        
        if msg_type == 'final_response':
            return data.get("message", "")
        elif msg_type == 'tool_call_required':
            return "[System] Yêu cầu gọi Tool (Chưa implement xử lý tool)"
        elif msg_type == 'error':
            print(f" Timely Logic Error: {data.get('error')}")
            return None
        return str(data)

    def call_api(self, prompt):

        try:
            self._ensure_auth()
            url = f"{self.base_url}/llm-completion"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.access_token}"
            }
            payload = self._build_payload(prompt)

            print(f"wb [SYNC CHAT] Sending: {prompt[:30]}...")
            resp = requests.post(url, json=payload, headers=headers)
            
            if resp.status_code == 401:
                self.access_token = None
                return self.call_api(prompt) # Retry once logic could be added here
            
            if resp.status_code not in [200, 201]:
                print(f" API Error {resp.status_code}: {resp.text}")
                return None

            return self._process_response(resp.json())

        except Exception as e:
            print(f" Sync Exception: {e}")
            return None

    async def call_api_async(self, prompt, retries=3):

        self._ensure_auth() 
        url = f"{self.base_url}/llm-completion"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        payload = self._build_payload(prompt)

        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 429: # Rate Limit
                            print(f" [Async] Rate Limit. Wait {2**attempt}s...")
                            await asyncio.sleep(2 ** attempt)
                            continue
                        
                        if resp.status == 401:
                            print(" Token hết hạn. Reset token.")
                            self.access_token = None
                            self._ensure_auth() # Refresh sync (chấp nhận chặn 1 chút) hoặc cần logic async auth
                            headers["Authorization"] = f"Bearer {self.access_token}"
                            continue

                        if resp.status not in [200, 201]:
                            text = await resp.text()
                            print(f" Async API Error {resp.status}: {text}")
                            return None
                        
                        data = await resp.json()
                        return self._process_response(data)
            except Exception as e:
                print(f" Async Exception (Attempt {attempt+1}): {e}")
                await asyncio.sleep(1)
        
        return None

    def _build_payload(self, prompt):
        return {
            "session_id": self.session_id,
            "messages": [{"role": "user", "content": prompt}],
            "chat_model_node": {
                "model": self.model_name,
            },
            "chat_type": "DYNAMIC_CHAT",
            "stream": False,
            "locale": "vi"
        }

    def get_embedding(self, text):
        if self.embedder:
            return self.embedder.encode(text).tolist()
        return []