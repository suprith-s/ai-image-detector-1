from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import os
import io
import base64

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
model.classifier = nn.Linear(model.classifier.in_features, 2)

model.load_state_dict(torch.load("ai_detector.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        file = request.files['image']

        if file:
            # READ IMAGE IN MEMORY (NO SAVE)
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            img = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img)
                probs = F.softmax(outputs, dim=1)[0]

            ai_prob = float(probs[0]) * 100
            real_prob = float(probs[1]) * 100

            # CONVERT IMAGE TO BASE64
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            img_src = f"data:image/jpeg;base64,{img_base64}"

            result = {
            "ai": round(ai_prob, 2),
            "real": round(real_prob, 2),
            "label": "AI Generated 🤖" if ai_prob > real_prob else "Real Image 📸",
            "image": img_src
}
    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
