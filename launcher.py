import os
os.environ["HF_HUB_OFFLINE"] = "1"
from server import app
app.run(debug=False, host="0.0.0.0", port=5000)
