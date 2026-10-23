import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image, ImageDraw
import vtracer
import tempfile
import io
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader

# ==========================================
# KDP FULL-BLEED CONSTANTS (8.625" x 11.25" @ 400 DPI)
# ==========================================
PAGE_W_IN = 8.625
PAGE_H_IN = 11.25
DPI = 400

TARGET_W = int(PAGE_W_IN * DPI)        # 3450 px
TARGET_H = int(PAGE_H_IN * DPI)        # 4500 px
BLEED_PX = int(0.125 * DPI)            # 50 px (0.125" bleed)
SAFE_MARGIN_PX = int(0.50 * DPI)       # 200 px (0.125" bleed + 0.375" safe margin)

st.set_page_config(page_title="KDP 400-DPI Coloring Book Studio", layout="wide")

st.title("🎨 KDP 400-DPI Coloring Book Pre-Press Studio")
st.caption(f"Full-Bleed Spec: {PAGE_W_IN}\" × {PAGE_H_IN}\" ({TARGET_W} × {TARGET_H} px @ {DPI} DPI)")


def fit_and_position_canvas(pil_img, fit_mode, zoom_pct, shift_x_pct, shift_y_pct):
    """Fits, scales, and shifts the image onto a pure white 3450x4500 canvas."""
    src_w, src_h = pil_img.size
    target_ratio = TARGET_W / TARGET_H
    src_ratio = src_w / src_h

    if fit_mode == "Cover Full Bleed (Auto-Fill)":
        if src_ratio > target_ratio:
            base_h = TARGET_H
            base_w = int(TARGET_H * src_ratio)
        else:
            base_w = TARGET_W
            base_h = int(TARGET_W / src_ratio)
    elif fit_mode == "Contain Inside Canvas (Pad with White)":
        if src_ratio > target_ratio:
            base_w = TARGET_W
            base_h = int(TARGET_W / src_ratio)
        else:
            base_h = TARGET_H
            base_w = int(TARGET_H * src_ratio)
    else:
        base_w, base_h = TARGET_W, TARGET_H

    scale = zoom_pct / 100.0
    new_w = max(1, int(base_w * scale))
    new_h = max(1, int(base_h * scale))

    resized_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas_img = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))

    offset_x = int((TARGET_W - new_w) / 2 + (shift_x_pct / 100.0) * (TARGET_W / 2))
    offset_y = int((TARGET_H - new_h) / 2 + (shift_y_pct / 100.0) * (TARGET_H / 2))

    canvas_img.paste(resized_img, (offset_x, offset_y))
    return canvas_img


# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2 = st.tabs(["🖌️ 1. Page Editor, Upscaler & Vectorizer", "📚 2. Batch KDP Interior PDF Compiler"])

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded_file = st.file_uploader("Upload Raw AI Coloring Page (PNG/JPG/WEBP)", type=["png", "jpg", "jpeg", "webp"])

        if uploaded_file is not None:
            raw_pil = Image.open(uploaded_file).convert("RGB")

            st.subheader("1. Spot-Eraser & Touch-Up Brush")
            brush_col1, brush_col2, brush_col3 = st.columns(3)
            with brush_col1:
                enable_brush = st.checkbox("Enable Brush Canvas", value=True)
            with brush_col2:
                brush_color = st.selectbox("Brush Ink", ["White (Erase #FFFFFF)", "Black (Draw #000000)"])
            with brush_col3:
                brush_size = st.slider("Brush Size", 2, 80, 20)

            stroke_hex = "#FFFFFF" if "White" in brush_color else "#000000"

            # Scale image for comfortable browser drawing
            disp_w = 520
            disp_h = int(disp_w * (raw_pil.height / raw_pil.width))

            if enable_brush:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0.0)",
                    stroke_width=brush_size,
                    stroke_color=stroke_hex,
                    background_image=raw_pil,
                    update_streamlit=True,
                    height=disp_h,
                    width=disp_w,
                    drawing_mode="freedraw",
                    key="touchup_canvas",
                )
                # Merge brush strokes onto original full-res image
                if canvas_result.image_data is not None:
                    overlay_rgba = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
                    overlay_resized = overlay_rgba.resize(raw_pil.size, Image.Resampling.NEAREST)
                    working_pil = raw_pil.copy().convert("RGBA")
                    working_pil.alpha_composite(overlay_resized)
                    working_pil = working_pil.convert("RGB")
                else:
                    working_pil = raw_pil
            else:
                working_pil = raw_pil
                st.image(raw_pil, use_container_width=True)

            st.subheader("2. Aspect-Ratio & Safe-Zone Positioning")
            fit_mode = st.radio(
                "Aspect-Ratio Fit Mode",
                ["Cover Full Bleed (Auto-Fill)", "Contain Inside Canvas (Pad with White)", "Stretch to Exact Dimensions"],
                horizontal=True
            )
            z_col, x_col, y_col = st.columns(3)
            with z_col:
                zoom_pct = st.slider("Zoom (%)", 50, 150, 100)
            with x_col:
                shift_x_pct = st.slider("Shift Left/Right (%)", -50, 50, 0)
            with y_col:
                shift_y_pct = st.slider("Shift Up/Down (%)", -50, 50, 0)

            st.subheader("3. Line Art Crispness & Cleanup")
            threshold_val = st.slider("Black/White Cutoff (Higher = Thicker/Darker Capture)", 80, 240, 185)
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                line_smoothing = st.slider("Curve Smoothing", 0, 7, 3)
            with s_col2:
                speckle_removal = st.slider("Speckle Cleanup", 0, 6, 2)
            with s_col3:
                line_weight = st.slider("Line Weight Tweak", -3, 3, 0)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                show_guides = st.checkbox("Show KDP Trim (Red) & Safe Zone (Blue)", value=True)
            with g_col2:
                generate_svg = st.checkbox("Also Generate Vector .SVG (for Inkscape)", value=False)

    with col_right:
        if uploaded_file is not None:
            st.subheader("4. 400-DPI KDP Master Preview & Export")

            with st.spinner("Processing 3,450 × 4,500 px @ 400 DPI..."):
                # 1. Fit & Position
                positioned_pil = fit_and_position_canvas(working_pil, fit_mode, zoom_pct, shift_x_pct, shift_y_pct)

                # 2. OpenCV Grayscale + Smoothing + Threshold
                img_np = np.array(positioned_pil)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

                if line_smoothing > 0:
                    ksize = int(line_smoothing) * 2 + 1
                    gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

                _, binary = cv2.threshold(gray, int(threshold_val), 255, cv2.THRESH_BINARY)

                if speckle_removal > 0:
                    k = int(speckle_removal) * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                if line_weight != 0:
                    k = abs(int(line_weight)) + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                    if line_weight > 0:
                        binary = cv2.erode(binary, kernel, iterations=1)
                    else:
                        binary = cv2.dilate(binary, kernel, iterations=1)

                final_pil = Image.fromarray(binary).convert("L")

                # Prepare 400 DPI PNG in memory
                png_buffer = io.BytesIO()
                final_pil.save(png_buffer, format="PNG", dpi=(DPI, DPI))
                png_bytes = png_buffer.getvalue()

                # Build Preview with KDP Guides
                preview_pil = final_pil.convert("RGB")
                if show_guides:
                    draw = ImageDraw.Draw(preview_pil)
                    draw.rectangle(
                        [BLEED_PX, BLEED_PX, TARGET_W - BLEED_PX, TARGET_H - BLEED_PX],
                        outline="#FF1744", width=8
                    )
                    draw.rectangle(
                        [SAFE_MARGIN_PX, SAFE_MARGIN_PX, TARGET_W - SAFE_MARGIN_PX, TARGET_H - SAFE_MARGIN_PX],
                        outline="#2979FF", width=8
                    )

                st.image(preview_pil, caption="Red = 0.125\" KDP Guillotine Trim | Blue = 0.375\" Safe Margin", use_container_width=True)

                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        label="📥 Download 400 DPI Print-Ready PNG",
                        data=png_bytes,
                        file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_KDP_400DPI.png",
                        mime="image/png",
                        use_container_width=True
                    )

                with dl_col2:
                    if generate_svg:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_png:
                            tmp_png.write(png_bytes)
                            tmp_png_path = tmp_png.name
                        tmp_svg_path = tmp_png_path.replace(".png", ".svg")

                        vtracer.convert_image_to_svg_py(
                            tmp_png_path,
                            tmp_svg_path,
                            colormode="binary",
                            hierarchical="stacked",
                            mode="spline",
                            filter_speckle=int(max(2, speckle_removal * 2)),
                            corner_threshold=60,
                            length_threshold=4.0,
                            max_iterations=10,
                            splice_threshold=45,
                            path_precision=3
                        )
                        with open(tmp_svg_path, "rb") as f_svg:
                            svg_bytes = f_svg.read()

                        st.download_button(
                            label="📥 Download Vector .SVG (For Inkscape)",
                            data=svg_bytes,
                            file_name=f"{uploaded_file.name.rsplit('.', 1)[0]}_Vector.svg",
                            mime="image/svg+xml",
                            use_container_width=True
                        )
        else:
            st.info("👈 Upload a raw AI coloring page on the left to start editing and upscaling.")

# ==========================================
# TAB 2: BATCH KDP PDF COMPILER
# ==========================================
with tab2:
    st.subheader("Compile Finished 400-DPI PNGs into a Single KDP Interior PDF")
    st.write("Upload your finished `3,450 × 4,500 px` PNG pages below to lock them into an exact **8.625\" × 11.25\"** full-bleed PDF.")

    batch_files = st.file_uploader("Select Finished 400-DPI PNG Pages", type=["png"], accept_multiple_files=True)
    insert_blank_backs = st.checkbox("Insert Blank Reverse Page After Every Illustration (Prevents Marker Bleed-Through)", value=True)

    if batch_files and st.button("📄 Build KDP Print-Ready PDF", type="primary"):
        with st.spinner("Compiling full-bleed KDP PDF..."):
            pdf_buffer = io.BytesIO()
            page_w_pts = PAGE_W_IN * 72.0
            page_h_pts = PAGE_H_IN * 72.0

            c = pdf_canvas.Canvas(pdf_buffer, pagesize=(page_w_pts, page_h_pts))
            for uploaded_page in batch_files:
                img = Image.open(uploaded_page).convert("RGB")
                c.drawImage(ImageReader(img), 0, 0, width=page_w_pts, height=page_h_pts)
                c.showPage()
                if insert_blank_backs:
                    c.showPage()
            c.save()

            st.success(f"Compiled {len(batch_files)} coloring pages into a KDP-ready interior PDF!")
            st.download_button(
                label="📥 Download Final KDP Interior PDF",
                data=pdf_buffer.getvalue(),
                file_name="KDP_Coloring_Book_Interior_FullBleed.pdf",
                mime="application/pdf"
            )
          
