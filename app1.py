import streamlit as st
from openai import OpenAI
import pandas as pd
import io
import base64
import json
import re
from PIL import Image  # NEW: for tiling
import math
import os
from dotenv import load_dotenv

# ----------------------------------------------------
# 
# ----------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Environment variable OPENAI_API_KEY is not set. Set it and rerun the app.")
    st.stop()
# ----------------------------------------------------



client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Legend → Plan Tag Counter", layout="wide")
st.title("Legend → Plan Tag Counter (GPT-5.1 Vision Based)")

legend_file = st.file_uploader("Upload Legend Image", type=["png","jpg","jpeg"])
plan_file = st.file_uploader("Upload Plan Image", type=["png","jpg","jpeg"])

def extract_text_from_image(image_bytes_b64: str, prompt: str):
    """
    Calls GPT-5-mini Vision to extract text from an image.
    """
    response = client.responses.create(
        model="gpt-5.1",
        input=[
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_bytes_b64}"
                    }
                ]
            }
        ]
    )
    return response.output_text

# -----------------------------
# NEW: Robust JSON parsing
# -----------------------------
def safe_json_array(raw: str):
    """
    Try to parse a top-level JSON array from model output.
    Falls back to extracting the first [...] block if needed.
    """
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Model output is not a JSON array.")

# -----------------------------
# NEW: Tiling helper
# -----------------------------
def tile_image_b64(image_b64: str, grid_size: int =5, overlap_px: int = 0):
    """
    Split base64-encoded image into grid_size x grid_size tiles with overlap.
    Returns a list of tile images encoded as base64 PNG strings.
    """
    # Decode base64 to PIL Image
    img_bytes = base64.b64decode(image_b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W, H = img.size

    # Compute nominal tile width/height
    tile_w = math.ceil(W / grid_size)
    tile_h = math.ceil(H / grid_size)

    tiles_b64 = []
    for gy in range(grid_size):  # vertical (rows)
        for gx in range(grid_size):  # horizontal (cols)
            left = max(0, gx * tile_w - overlap_px)
            top = max(0, gy * tile_h - overlap_px)
            right = min(W, (gx + 1) * tile_w + overlap_px)
            bottom = min(H, (gy + 1) * tile_h + overlap_px)
            crop = img.crop((left, top, right, bottom))

            # Encode cropped tile to base64 PNG
            buf = io.BytesIO()
            crop.save(buf, format="PNG", optimize=True)
            buf.seek(0)
            tile_b64 = base64.b64encode(buf.read()).decode()
            tiles_b64.append(tile_b64)
    return tiles_b64

if legend_file and plan_file:

    # Convert both images to base64
    legend_bytes_b64 = base64.b64encode(legend_file.read()).decode()
    plan_bytes_b64 = base64.b64encode(plan_file.read()).decode()

    # -------------------------------------
    # STEP 1 — Extract TAGS from legend
    # -------------------------------------
    st.subheader("Step 1 — Extracting tags from legend…")

    legend_prompt = """
    Extract ALL TAG values that appear in this legend table.
    A TAG is typically something like F-01,F-02,F-03,X1,X2, D-01,D-02,F-08A,F-08B, etc.

    Return ONLY a JSON list of tags.
    Example:
    ["F-01", "F-02", "F-03", "X1","X2","F-08A"]

    No descriptions.
    No symbols.
    No extra text.
    """

    legend_raw = extract_text_from_image(legend_bytes_b64, legend_prompt)

    try:
        legend_tags = safe_json_array(legend_raw)  # updated parser
    except Exception:
        st.error("Unable to parse TAGs from legend. GPT returned:")
        st.write(legend_raw)
        st.stop()

    st.success("Legend TAGs extracted successfully.")
    st.write("**Legend TAGs:**", legend_tags)

    # -------------------------------------
    # NEW: Tiling UI for plan OCR
    # -------------------------------------
    st.subheader("Step 2 — Extracting all text from plan…")
    enable_tiling = st.checkbox("Enable tiling for plan OCR (recommended for large plans)", value=True)
    grid_size = st.slider("Grid size (N×N)", min_value=1, max_value=20, value=5, help="1 = no tiling; 3 = 3×3 tiles")
    overlap_px = st.slider("Tile overlap (px)", min_value=0, max_value=256, value=0, help="Helps capture text near tile edges")

    plan_prompt = """
    Extract ALL textual elements from this architectural plan image.
    Return ONLY a JSON list of text tokens found.
    Preserve duplicates.

    Example:
    ["F-01", "F-02", "F-03", "X1","X2","F-08A","F-08B"]
    """

    # -------------------------------------
    # STEP 2 — Extract tokens (with optional tiling)
    # -------------------------------------
    plan_tokens = []
    try:
        if enable_tiling and grid_size > 1:
            with st.spinner(f"Using {grid_size}×{grid_size} tiling with {overlap_px}px overlap…"):
                tiles_b64 = tile_image_b64(plan_bytes_b64, grid_size=grid_size, overlap_px=overlap_px)
                st.caption(f"Processing {len(tiles_b64)} tiles…")
                for idx, tile_b64 in enumerate(tiles_b64, start=1):
                    raw = extract_text_from_image(tile_b64, plan_prompt)
                    tokens = safe_json_array(raw)
                    plan_tokens.extend(tokens)  # keep duplicates
            st.success(f"Plan text extraction completed across {grid_size*grid_size} tiles.")
        else:
            with st.spinner("Processing full plan image…"):
                plan_raw = extract_text_from_image(plan_bytes_b64, plan_prompt)
                plan_tokens = safe_json_array(plan_raw)
            st.success("Plan text extraction completed successfully.")
    except Exception as e:
        st.error(f"Unable to parse text from plan image: {e}")
        st.stop()

    st.write("**Sample extracted tokens:**", plan_tokens[:30])

    # -------------------------------------
    # STEP 3 — Count tag occurrences
    # -------------------------------------
    st.subheader("Step 3 — Counting tag occurrences…")

    counts = {}
    for tag in legend_tags:
        counts[tag] = sum(1 for t in plan_tokens if t == tag)

    df = pd.DataFrame(list(counts.items()), columns=["Tag", "Count"])
    st.table(df)

    # -------------------------------------
    # Excel Download
    # -------------------------------------
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="openpyxl") as writer:  # engine specified
        df.to_excel(writer, index=False, sheet_name="Tag Counts")
    towrite.seek(0)

    st.download_button(
        label="Download Tag Counts (Excel)",
        data=towrite,
        file_name="tag_counts.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("Upload both Legend and Plan images to begin processing.")