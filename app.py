import streamlit as st
import torch
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="ConvNeXtV2 – HyperKvasir",
    layout="centered"
)

NUM_CLASSES = 12
MODEL_PATH = "model/convnextv2_hyperkvasir.pth"

# STREAMLIT CLOUD = CPU ONLY
device = torch.device("cpu")

# =========================================================
# LOAD MODEL (CACHE)
# =========================================================
@st.cache_resource
def load_model():
    model = timm.create_model(
    "convnextv2_tiny",
    pretrained=False,
    num_classes=NUM_CLASSES
)


    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model


# =========================================================
# LOAD MODEL
# =========================================================
st.title("Klasifikasi citra endoskopi")

try:
    model = load_model()
    st.success("Model berhasil dimuat & siap digunakan ✅")
    st.write("**Arsitektur:** ConvNeXtV2 Tiny")
    st.write("**Jumlah kelas:**", NUM_CLASSES)
except Exception as e:
    st.error("Gagal memuat model ❌")
    st.exception(e)
    st.stop()

# =========================================================
# INFERENCE TRANSFORM
# =========================================================
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# CLASS NAMES (HARUS SAMA DENGAN TRAINING)
# =========================================================
class_names = [
    "bbps-0-1",
    "bbps-2-3",
    "cecum",
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "polyps",
    "pylorus",
    "retroflex-rectum",
    "retroflex-stomach",
    "ulcerative-colitis",
    "z-line"
]

# =========================================================
# UI UPLOAD & PREDICTION
# =========================================================
st.markdown("---")
st.markdown("## 📤 Upload Gambar Endoskopi")

uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar Input",
        use_container_width=True
    )

    input_tensor = inference_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_idx.item()]

    st.markdown("---")
    st.subheader("🩺 Hasil Prediksi")

    st.success(f"**Diagnosis:** {predicted_class}")
    st.info(f"**Confidence:** {confidence.item() * 100:.2f}%")

    # ===============================
    # TOP-3 PROBABILITIES
    # ===============================
    st.markdown("### 🔝 Top-3 Prediksi")
    top_probs, top_idxs = torch.topk(probabilities, 3)

    for i in range(3):
        st.write(
            f"{i+1}. **{class_names[top_idxs[0][i]]}** "
            f"({top_probs[0][i].item() * 100:.2f}%)"
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("ConvNeXtV2 – HyperKvasir | Klasifikasi citra endoskopi gastrointestinal")
