from fastapi import FastAPI, UploadFile, File, Query
from PIL import Image
import base64
from io import BytesIO

from enhancer import enhance_image
from ocr_engine import extract_text
from doc_classifier import classify_document

app = FastAPI()

def image_to_base64(pil_image):
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/process/")
async def process(
    file: UploadFile = File(...),

    # OCR params
    text_threshold: float = Query(0.6),
    low_text: float = Query(0.2),
    link_threshold: float = Query(0.2),
):
    
    contents = await file.read()

    # Document classification (RAW IMAGE ✅)
    document = classify_document(contents)

    # 1️⃣ Load RAW image
    raw_image = Image.open(BytesIO(contents))
    

    # 2️⃣ Enhance image (FOR OCR ONLY)
    enhanced_image = enhance_image(raw_image, outscale=2)

    # 3️⃣ Document classification
    #document = classify_document(enhanced_image)

    # 4️⃣ OCR on enhanced image
    ocr = extract_text(
        enhanced_image,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
    )

    # 5️⃣ Base64 preview
    enhanced_image_base64 = image_to_base64(enhanced_image)

    return {
        "status": "success",
        "document": document,
        "ocr": ocr,
        "enhanced_image_base64": enhanced_image_base64
    }
