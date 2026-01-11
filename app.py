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
    "bbps-0-1": (
        "Skor Boston Bowel Preparation Scale (BBPS) 0–1 menunjukkan "
        "persiapan usus yang buruk, sehingga visualisasi mukosa kolon "
        "tidak optimal."
    ),
    "bbps-2-3": (
        "Skor BBPS 2–3 menunjukkan persiapan usus yang adekuat hingga baik, "
        "memungkinkan visualisasi mukosa kolon dengan jelas."
    ),
    "cecum": (
        "Cecum merupakan bagian awal usus besar dan menjadi landmark penting "
        "untuk memastikan kolonoskopi telah mencapai area maksimal."
    ),
    "dyed-lifted-polyps": (
        "Polip yang telah diwarnai dan diangkat menggunakan teknik injeksi "
        "submukosa untuk membantu reseksi yang aman."
    ),
    "dyed-resection-margins": (
        "Area tepi reseksi yang diwarnai untuk mengevaluasi batas jaringan "
        "dan memastikan lesi telah diangkat sepenuhnya."
    ),
    "esophagitis": (
        "Esofagitis adalah peradangan pada mukosa esofagus, sering kali "
        "disebabkan oleh refluks asam lambung."
    ),
    "polyps": (
        "Polip merupakan pertumbuhan jaringan abnormal pada mukosa saluran "
        "cerna yang berpotensi menjadi ganas."
    ),
    "pylorus": (
        "Pilorus adalah bagian distal lambung yang menghubungkan lambung "
        "dengan duodenum dan mengatur pengosongan lambung."
    ),
    "retroflex-rectum": (
        "Tampilan retrofleksi pada rektum digunakan untuk mendeteksi lesi "
        "yang sulit terlihat pada pandangan standar."
    ),
    "retroflex-stomach": (
        "Tampilan retrofleksi pada lambung membantu visualisasi area "
        "fundus dan kardia."
    ),
    "ulcerative-colitis": (
        "Ulcerative colitis merupakan penyakit inflamasi usus kronis "
        "yang ditandai peradangan dan ulserasi pada kolon."
    ),
    "z-line": (
        "Z-line adalah batas antara epitel esofagus dan lambung, "
        "penting dalam evaluasi GERD."
    ),
}

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

    # =====================================================
    # RESULT
    # =====================================================
    st.markdown("---")
    st.subheader("🩺 Hasil Prediksi")

    st.success(f"**Diagnosis:** {predicted_class}")
    st.info(f"**Confidence:** {confidence.item() * 100:.2f}%")

    # =====================================================
    # CLASS EXPLANATION
    # =====================================================
    st.markdown("### 📚 Penjelasan Kelas")
    st.write(class_descriptions.get(
        predicted_class,
        "Tidak tersedia penjelasan untuk kelas ini."
    ))

    # =====================================================
    # TOP-3 PROBABILITIES
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
        "⚠️ Hasil ini merupakan sistem pendukung berbasis kecerdasan buatan "
        "dan **tidak menggantikan diagnosis klinis oleh dokter spesialis**."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "ConvNeXtV2 – HyperKvasir | Klasifikasi citra endoskopi gastrointestinal"
)
