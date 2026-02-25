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
CONFIDENCE_THRESHOLD = 0.60  # 60% threshold

device = torch.device("cpu")  # Streamlit Cloud CPU only

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
    "bbps-0-1": "Skor BBPS 0–1 menunjukkan persiapan usus yang buruk sehingga visualisasi mukosa kolon tidak optimal.",
    "bbps-2-3": "Skor BBPS 2–3 menunjukkan persiapan usus yang adekuat hingga baik dengan visualisasi mukosa yang jelas.",
    "cecum": "Cecum adalah bagian awal usus besar dan menjadi landmark penting dalam kolonoskopi.",
    "dyed-lifted-polyps": "Polip yang diwarnai dan diangkat menggunakan teknik injeksi submukosa.",
    "dyed-resection-margins": "Area tepi reseksi yang diwarnai untuk memastikan batas jaringan telah diangkat.",
    "esophagitis": "Peradangan pada mukosa esofagus yang sering disebabkan oleh refluks asam lambung.",
    "polyps": "Pertumbuhan jaringan abnormal pada mukosa saluran cerna yang berpotensi berkembang menjadi kanker.",
    "pylorus": "Bagian distal lambung yang menghubungkan lambung dengan duodenum.",
    "retroflex-rectum": "Tampilan retrofleksi rektum untuk mendeteksi lesi tersembunyi.",
    "retroflex-stomach": "Tampilan retrofleksi lambung untuk melihat area fundus dan kardia.",
    "ulcerative-colitis": "Penyakit inflamasi usus kronis yang ditandai peradangan dan ulserasi pada kolon.",
    "z-line": "Batas antara epitel esofagus dan lambung yang penting dalam evaluasi GERD."
}

# =========================================================
# UPLOAD UI
# =========================================================
st.markdown("---")
st.markdown("## 📤 Upload Gambar Endoskopi")

uploaded_file = st.file_uploader(
    "Upload gambar (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
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

    # =====================================================
    # RESULT
    # =====================================================
    st.markdown("---")
    st.subheader("🩺 Hasil Prediksi")

    if confidence_score >= CONFIDENCE_THRESHOLD:

        st.success(f"**Diagnosis:** {predicted_class}")
        st.info(f"Confidence: {confidence_score * 100:.2f}%")

        st.markdown("### 📚 Penjelasan Kelas")
        st.write(class_descriptions.get(
            predicted_class,
            "Tidak tersedia penjelasan."
        ))

    else:

        st.warning("⚠️ Model memiliki tingkat keyakinan rendah.")
        st.error(f"Prediksi sementara: {predicted_class}")
        st.info(f"Confidence rendah: {confidence_score * 100:.2f}%")

        st.markdown("### 🔍 Rekomendasi")
        st.write(
            "Hasil prediksi tidak cukup meyakinkan. "
            "Disarankan untuk evaluasi lanjutan oleh dokter spesialis."
        )

    # =====================================================
    # TOP 3
    # =====================================================
    st.markdown("### 🔝 Top-3 Prediksi")

    top_probs, top_idxs = torch.topk(probabilities, 3)

    for i in range(3):
        st.write(
            f"{i+1}. **{class_names[top_idxs[0][i]]}** "
            f"({top_probs[0][i].item() * 100:.2f}%)"
        )

    # =====================================================
    # DISCLAIMER
    # =====================================================
    st.warning(
        "⚠️ Sistem ini merupakan alat bantu berbasis kecerdasan buatan "
        "dan tidak menggantikan diagnosis klinis oleh tenaga medis profesional."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "ConvNeXtV2 – HyperKvasir | Sistem Klasifikasi Citra Endoskopi Gastrointestinal"
)
