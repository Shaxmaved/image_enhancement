import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'], gpu=False)

def extract_text(
    pil_image: Image.Image,
    text_threshold=0.6,
    low_text=0.2,
    link_threshold=0.2,
):
    img_np = np.array(pil_image)

    results = reader.readtext(
        img_np,
        detail=1,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
    )

    detections = []
    full_text = []

    for bbox, text, conf in results:
        clean_bbox = [[int(x), int(y)] for x, y in bbox]

        detections.append({
            "text": str(text),
            "confidence": float(conf),
            "bounding_box": clean_bbox
        })
        full_text.append(text)

    return {
        "full_text": "\n".join(full_text),
        "detections": detections
    }


# import easyocr
# import numpy as np
# from PIL import Image

# # Load OCR model ONCE
# reader = easyocr.Reader(
#     ['en'],
#     gpu=False
# )

# def extract_text(
#     pil_image: Image.Image,
#     text_threshold: float = 0.6,
#     low_text: float = 0.2,
#     link_threshold: float = 0.2,
#     #canvas_size: int = 2560,
#     #mag_ratio: float = 1.0
# ) -> str:

#     img_np = np.array(pil_image)

#     results = reader.readtext(
#         img_np,
#         detail=0,
#         text_threshold=text_threshold,
#         low_text=low_text,
#         link_threshold=link_threshold,
#         #canvas_size=canvas_size,
#         #mag_ratio=mag_ratio
#     )

#     return "\n".join(results)


# import easyocr
# import numpy as np
# from PIL import Image

# # Load OCR model once
# reader = easyocr.Reader(['en'], gpu=False)

# def extract_text(pil_image: Image.Image) -> str:
#     img_np = np.array(pil_image)
#     results = reader.readtext(img_np, detail=0)
#     return "\n".join(results)
