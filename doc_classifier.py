from ultralytics import YOLO

# =====================================================
# LOAD YOLO CLASSIFICATION MODEL (ONCE)
# =====================================================
model = YOLO("DocClasfnModel.pt")

print("✅ YOLO task detected:", model.task)  # should be 'classify'


# =====================================================
# DOCUMENT CLASSIFICATION FUNCTION (DO NOT MODIFY)
# =====================================================
def classify_document(image_bytes: bytes) -> dict:
    """
    Classify document type using RAW image bytes.
    This preserves original model accuracy.
    """

    # 🔥 IMPORTANT:
    # Pass image BYTES directly to YOLO
    # This ensures:
    # - Same preprocessing as training
    # - Correct resizing & normalization
    # - EXIF orientation preserved
    # - No color / dtype distortion

    results = model(image_bytes, verbose=False)

    # SAFETY CHECK
    if not results or results[0].probs is None:
        return {
            "document_type": "UNKNOWN",
            "confidence": 0.0
        }

    # TOP-1 prediction
    cls_id = int(results[0].probs.top1)
    confidence = float(results[0].probs.top1conf)

    label = model.names[cls_id]

    return {
        "document_type": str(label),
        "confidence": round(confidence, 3)
    }
