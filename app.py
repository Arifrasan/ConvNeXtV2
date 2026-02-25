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
CONFIDENCE_THRESHOLD = 0.60  # 60%

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
# TITLE
# =========================================================
st.title("Klasifikasi Citra Endoskopi Gastrointestinal")

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
# TRANSFORM
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
# CLASS NAMES
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
# CLASS DESCRIPTIONS
# =========================================================
class_descriptions = {
    "bbps-0-1": "Skor BBPS 0–1 menunjukkan persiapan usus yang buruk.",
    "bbps-2-3": "Skor BBPS 2–3 menunjukkan persiapan usus yang adekuat.",
    "cecum": "Cecum adalah bagian awal usus besar.",
    "dyed-lifted-polyps": "Polip yang diwarnai dan diangkat.",
    "dyed-resection-margins": "Area tepi reseksi yang diwarnai.",
    "esophagitis": "Peradangan pada mukosa esofagus.",
    "polyps": "Pertumbuhan jaringan abnormal pada mukosa.",
    "pylorus": "Bagian distal lambung menuju duodenum.",
    "retroflex-rectum": "Tampilan retrofleksi pada rektum.",
    "retroflex-stomach": "Tampilan retrofleksi pada lambung.",
    "ulcerative-colitis": "Penyakit inflamasi usus kronis.",
    "z-line": "Batas antara esofagus dan lambung."
}

# =========================================================
# UPLOAD UI
# =========================================================
st.markdown("---")
st.markdown("## 📤 Upload Gambar Endoskopi")

uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png", "jfif"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Input", use_container_width=True)

    input_tensor = inference_transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_idx.item()]
    confidence_score = confidence.item()

    st.markdown("---")
    st.subheader("🩺 Hasil Analisis")

    # =====================================================
    # STRICT LOW CONFIDENCE HANDLING
    # =====================================================
    if confidence_score >= CONFIDENCE_THRESHOLD:

        st.success(f"**Diagnosis:** {predicted_class}")
        st.info(f"Confidence: {confidence_score * 100:.2f}%")

        st.markdown("### 📚 Penjelasan Kelas")
        st.write(class_descriptions.get(
            predicted_class,
            "Tidak tersedia penjelasan."
        ))

        # TOP-3 hanya tampil jika confidence cukup
        st.markdown("### 🔝 Top-3 Prediksi")
        top_probs, top_idxs = torch.topk(probabilities, 3)

        for i in range(3):
            st.write(
                f"{i+1}. **{class_names[top_idxs[0][i]]}** "
                f"({top_probs[0][i].item() * 100:.2f}%)"
            )

    else:

        st.error("❌ Sistem tidak dapat memberikan hasil prediksi. Gambar yang diupload kurang jelas, atau sistem tidak dapat mengidentifikasi citra karena berada di luar domain pelatihan model.")


        st.markdown("### 🔍 Rekomendasi")
        st.write(
            "Disarankan untuk menggunakan gambar dengan kualitas lebih baik "
            "atau melakukan evaluasi langsung oleh dokter spesialis."
        )

    # =====================================================
    # DISCLAIMER
    # =====================================================
    st.warning(
        "⚠️ Sistem ini merupakan alat bantu berbasis kecerdasan buatan "
        "dan tidak menggantikan diagnosis klinis profesional."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "ConvNeXtV2 – HyperKvasir | Sistem Klasifikasi Citra Endoskopi Gastrointestinal"
)
