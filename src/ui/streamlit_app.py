from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from PIL import Image

from src.core.config import CONFIG
from src.core.data import VerificationPair, parse_verification_pairs
from src.ml.emotion import EmotionDetector
from src.ml.liveness import LivenessDetector
from src.ml.verification import FaceVerifier


def load_registry() -> list[dict[str, Any]]:
    registry_file = CONFIG.registry_file
    if not registry_file.exists():
        return []
    return json.loads(registry_file.read_text(encoding="utf-8"))


def save_registry(entries: list[dict[str, Any]]) -> None:
    CONFIG.registry_file.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.registry_file.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def register_identity(name: str, image_path: Path) -> None:
    entries = load_registry()
    entries.append({"name": name, "image_path": str(image_path)})
    save_registry(entries)


def build_verifier() -> FaceVerifier | None:
    candidates = [
        CONFIG.checkpoints_dir / "best_embedding_metric_local.pt",
        CONFIG.checkpoints_dir / "best_embedding_classification_local.pt",
        CONFIG.checkpoints_dir / "embedding_metric_local.pt",
        CONFIG.checkpoints_dir / "embedding_classification_local.pt",
        CONFIG.checkpoints_dir / "embedding_classification.pt",
        CONFIG.checkpoints_dir / "embedding_metric_learning.pt",
    ]
    for checkpoint_path in candidates:
        if checkpoint_path.exists():
            return FaceVerifier.from_checkpoint(checkpoint_path=checkpoint_path, embedding_dim=CONFIG.embedding_dim)
    return None


def main() -> None:
    st.set_page_config(page_title="Face Verification Demo", layout="wide")
    st.title("Facial Recognition with Emotion and Liveness")
    st.write("Upload face images for verification, and register identities locally.")

    verifier = build_verifier()
    liveness_detector = LivenessDetector(CONFIG.checkpoints_dir / "liveness_model.pt")
    emotion_detector = EmotionDetector(CONFIG.checkpoints_dir / "emotion_model.pt")

    # ==== SECTION 1: Compare two faces (Face Verification) - Available immediately ====
    st.subheader("Compare two faces (Face Verification)")
    st.write("Upload two face images to test the trained verification model.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Face A**")
        uploaded_a = st.file_uploader("Choose first image", type=["jpg", "jpeg", "png"], key="face_a")
    with col_b:
        st.write("**Face B**")
        uploaded_b = st.file_uploader("Choose second image", type=["jpg", "jpeg", "png"], key="face_b")
    
    if uploaded_a is not None and uploaded_b is not None:
        image_a = Image.open(uploaded_a).convert("RGB")
        image_b = Image.open(uploaded_b).convert("RGB")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(image_a, caption="Face A", width='stretch')
        with col_b:
            st.image(image_b, caption="Face B", width='stretch')
        
        if verifier is None:
            st.error("Verification model not loaded. Cannot compare faces.")
        else:
            try:
                # Compute embeddings
                from src.training.common import build_transforms
                _, eval_tf = build_transforms(CONFIG.image_size)
                
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                img_a_tensor = eval_tf(image_a).unsqueeze(0).to(device)
                img_b_tensor = eval_tf(image_b).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    emb_a = verifier.model(img_a_tensor).cpu()
                    emb_b = verifier.model(img_b_tensor).cpu()
                
                # Compute cosine similarity
                from torch.nn.functional import cosine_similarity
                similarity = cosine_similarity(emb_a, emb_b, dim=1).item()
                
                # Threshold for decision (conservative threshold for face verification)
                threshold = 0.75
                prediction = "SAME PERSON" if similarity > threshold else "DIFFERENT PERSONS"
                pred_color = "green" if similarity > threshold else "red"
                
                st.markdown(f"### Verification Result")
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.metric("Similarity Score", f"{similarity:.4f}")
                with col_result2:
                    st.metric("Decision Threshold", f"{threshold:.2f}")
                
                st.markdown(f"<h3 style='color: {pred_color};'>{prediction}</h3>", unsafe_allow_html=True)
                st.info(f"Similarity score: {similarity:.4f} (threshold: {threshold:.2f})")
                
            except Exception as e:
                st.error(f"Error during verification: {e}")
    
    # ==== SECTION 2: Single image liveness/emotion analysis ====
    st.divider()
    st.subheader("Single image analysis (Liveness & Emotion)")
    st.write("Upload one image to check liveness and emotion detection.")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="single_image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input image", width='stretch')

        live_result = liveness_detector.predict(image)
        emotion_result = emotion_detector.predict(image)

        col1, col2, col3 = st.columns(3)
        col1.metric("Liveness", live_result.label)
        col2.metric("Emotion", emotion_result.label)
        col3.metric("Emotion confidence", f"{emotion_result.confidence:.2f}")

        if verifier is None:
            st.warning("No verification checkpoint found yet. Add a trained model to checkpoints/ to enable face matching.")
        else:
            st.success("Verification model loaded.")

        st.subheader("Register identity")
        identity_name = st.text_input("New identity name")
        if st.button("Save uploaded face"):
            if not identity_name.strip():
                st.error("Enter an identity name first.")
            else:
                registry_dir = CONFIG.data_dir / "registered_faces"
                registry_dir.mkdir(parents=True, exist_ok=True)
                image_path = registry_dir / f"{identity_name.strip()}_{len(load_registry()) + 1}.png"
                image.save(image_path)
                register_identity(identity_name.strip(), image_path)
                st.success(f"Saved {identity_name.strip()} to the local registry.")

        st.subheader("Local registry")
        registry_entries = load_registry()
        if registry_entries:
            st.json(registry_entries)
        else:
            st.caption("No registered identities yet.")
    
    # ==== SECTION 3: Validation pairs preview ====
    st.divider()
    st.subheader("Validation pairs preview")
    val_pairs_file = CONFIG.data_dir / "verification_pairs_val.txt"
    if val_pairs_file.exists():
        preview_pairs = parse_verification_pairs(val_pairs_file)[:5]
        st.table([
            {"image_a": pair.image_a, "image_b": pair.image_b, "label": pair.label}
            for pair in preview_pairs
        ])


if __name__ == "__main__":
    main()
