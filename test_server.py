import os, subprocess, time, json, urllib.request

os.environ["HF_HUB_OFFLINE"] = "1"

proc = subprocess.Popen(
    ["venv/Scripts/python.exe", "-c", "import os; os.environ['HF_HUB_OFFLINE']='1'; from server import app; app.run(debug=False, port=5000)"],
    cwd=os.path.dirname(os.path.abspath(__file__)),
)

time.sleep(15)

try:
    req = urllib.request.Request("http://127.0.0.1:5000/")
    resp = urllib.request.urlopen(req, timeout=10)
    print(f"GET /: {resp.status}")

    body = json.dumps({"session_id": "test", "message": "What is this document about?"}).encode()
    req2 = urllib.request.Request(
        "http://127.0.0.1:5000/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp2 = urllib.request.urlopen(req2, timeout=60)
    result = json.loads(resp2.read())
    print(f"Answer: {result.get('answer', '')[:200]}")
except Exception as e:
    print(f"Error: {e}")
finally:
    proc.terminate()
    proc.wait(timeout=5)
