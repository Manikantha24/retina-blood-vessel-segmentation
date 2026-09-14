import os
import time
import base64
import cv2
import gradio as gr
import numpy as np
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download


# ============================================================
# RETINA VISION AI
# Production-style Gradio application
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILENAME = "unet_resnet34_tversky_bce35.pth"
MODEL_REPO = "manni24/retina-vision-ai-model"
LOCAL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    MODEL_FILENAME
)


def get_model_path():
    # Use the local checkpoint during development when available.
    if os.path.exists(LOCAL_MODEL_PATH):
        return LOCAL_MODEL_PATH

    # Render / cloud deployment: download the checkpoint from Hugging Face.
    return hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )


MODEL_PATH = get_model_path()

SAMPLE_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "demo_retina.png"
)

THRESHOLD = 0.6


# ============================================================
# CUSTOM MEDICAL AI LOGO
# ============================================================

LOGO = """
<svg
    width="62"
    height="62"
    viewBox="0 0 62 62"
    xmlns="http://www.w3.org/2000/svg"
>

    <rect
        x="1"
        y="1"
        width="60"
        height="60"
        rx="18"
        fill="#071D29"
        stroke="#00D9D9"
        stroke-opacity="0.55"
    />

    <!-- Eye -->
    <ellipse
        cx="31"
        cy="31"
        rx="19"
        ry="12"
        fill="none"
        stroke="#00E5E5"
        stroke-width="1.8"
    />

    <!-- Iris -->
    <circle
        cx="31"
        cy="31"
        r="6.5"
        fill="none"
        stroke="#00E5E5"
        stroke-width="1.8"
    />

    <!-- Center -->
    <circle
        cx="31"
        cy="31"
        r="2.5"
        fill="#29F2E6"
    />

    <!-- Vessel branches -->
    <path
        d="M12 31H19"
        stroke="#00E5E5"
        stroke-width="1.5"
    />

    <path
        d="M43 31H50"
        stroke="#00E5E5"
        stroke-width="1.5"
    />

    <path
        d="M31 24V18"
        stroke="#00E5E5"
        stroke-width="1.5"
    />

    <path
        d="M31 38V44"
        stroke="#00E5E5"
        stroke-width="1.5"
    />

    <circle
        cx="19"
        cy="31"
        r="1.8"
        fill="#29F2E6"
    />

    <circle
        cx="43"
        cy="31"
        r="1.8"
        fill="#29F2E6"
    />

</svg>
"""


# ============================================================
# LOAD HERO RETINA IMAGE AS BASE64
# ============================================================

def get_image_data_uri(path):

    if not os.path.exists(path):
        return ""

    try:
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(
                image_file.read()
            ).decode("utf-8")

        return f"data:image/png;base64,{encoded}"

    except Exception:
        return ""


HERO_IMAGE_URI = get_image_data_uri(
    SAMPLE_PATH
)


# ============================================================
# MODEL
# ============================================================

print("=" * 46)
print("RETINA VISION AI")
print("=" * 46)
print("Loading trained model...")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint
)

model.to(device)
model.eval()

print(f"Device: {device}")
print(f"Model: {MODEL_PATH}")
print("=" * 46)
print("Model loaded successfully.")
print(f"Threshold: {THRESHOLD}")
print("=" * 46)


# ============================================================
# SAMPLE IMAGE
# ============================================================

def load_sample():

    if not os.path.exists(SAMPLE_PATH):
        return None

    image = cv2.imread(
        SAMPLE_PATH
    )

    if image is None:
        return None

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):

    if image is None:

        return (
            None,
            None,
            None,
            None,
            """
            <div class="analysis-error">
                ⚠ Please upload a retinal image.
            </div>
            """
        )

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Normalize input
    # --------------------------------------------------------

    original = np.asarray(
        image
    ).astype(
        np.uint8
    )

    if original.ndim == 2:

        original = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB
        )

    if original.shape[-1] == 4:

        original = original[:, :, :3]


    # --------------------------------------------------------
    # Resize for inference
    # --------------------------------------------------------

    resized = cv2.resize(
        original,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    image_tensor = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    image_tensor = torch.from_numpy(
        image_tensor
    ).permute(
        2,
        0,
        1
    ).unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            image_tensor
        )

        probability = torch.sigmoid(
            prediction
        )[0, 0].cpu().numpy()


    inference_time = (
        time.perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # Restore original resolution
    # --------------------------------------------------------

    probability_original = cv2.resize(
        probability,
        (
            original.shape[1],
            original.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # Binary segmentation
    # --------------------------------------------------------

    mask_binary = (
        probability_original
        >= THRESHOLD
    )

    mask = (
        mask_binary.astype(
            np.uint8
        ) * 255
    )


    # --------------------------------------------------------
    # Vessel overlay
    # --------------------------------------------------------

    vessel_color = np.zeros_like(
        original
    )

    vessel_color[:, :, 1] = 220
    vessel_color[:, :, 2] = 200

    blended = cv2.addWeighted(
        original,
        0.45,
        vessel_color,
        0.55,
        0
    )

    overlay = original.copy()

    overlay[mask_binary] = (
        blended[mask_binary]
    )


    # --------------------------------------------------------
    # Probability heatmap
    # --------------------------------------------------------

    heatmap_uint8 = (
        probability_original * 255
    ).astype(
        np.uint8
    )

    heatmap = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_TURBO
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Quantitative analysis
    # --------------------------------------------------------

    vessel_pixels = int(
        np.sum(mask_binary)
    )

    total_pixels = int(
        mask_binary.size
    )

    vessel_coverage = (
        vessel_pixels
        / total_pixels
    ) * 100


    # --------------------------------------------------------
    # Dynamic analysis status
    # --------------------------------------------------------

    status = f"""
    <div class="analysis-success">

        <div class="status-header">

            <div class="status-dot"></div>

            <div>
                <div class="status-title">
                    ANALYSIS COMPLETE
                </div>

                <div class="status-subtitle">
                    Vessel segmentation successfully generated
                </div>
            </div>

        </div>


        <div class="metric-grid">

            <div class="metric">

                <span class="metric-label">
                    Vessel Coverage
                </span>

                <span class="metric-value">
                    {vessel_coverage:.2f}%
                </span>

            </div>


            <div class="metric">

                <span class="metric-label">
                    Vessel Pixels
                </span>

                <span class="metric-value">
                    {vessel_pixels:,}
                </span>

            </div>


            <div class="metric">

                <span class="metric-label">
                    Inference Time
                </span>

                <span class="metric-value">
                    {inference_time:.2f}s
                </span>

            </div>


            <div class="metric">

                <span class="metric-label">
                    Threshold
                </span>

                <span class="metric-value">
                    {THRESHOLD}
                </span>

            </div>

        </div>

    </div>
    """

    return (
        original,
        mask,
        overlay,
        heatmap,
        status
    )


# ============================================================
# TRY SAMPLE
# ============================================================

def try_sample():

    sample = load_sample()

    if sample is None:

        return (
            None,
            None,
            None,
            None,
            """
            <div class="analysis-error">
                Sample retinal image could not be loaded.
            </div>
            """
        )

    return analyze_image(
        sample
    )


# ============================================================
# PREMIUM UI CSS
# ============================================================

CSS = f"""

/* =========================================================
   GLOBAL
   ========================================================= */

html,
body,
.gradio-container {{

    background:
        radial-gradient(
            circle at 12% 5%,
            rgba(0, 229, 229, 0.075),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 18%,
            rgba(0, 155, 185, 0.07),
            transparent 32%
        ),

        #020C14 !important;

    color:
        #DDF7F8 !important;

    font-family:
        Inter,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}}


/* =========================================================
   BRAND
   ========================================================= */

.brand-wrapper {{

    max-width:
        1120px;

    margin:
        0 auto;

    padding:
        28px 28px 20px;

    border-bottom:
        1px solid
        rgba(61, 220, 220, 0.13);
}}

.brand {{

    display:
        flex;

    align-items:
        center;

    gap:
        16px;
}}

.brand-logo {{

    width:
        62px;

    height:
        62px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    filter:
        drop-shadow(
            0 0 12px
            rgba(0, 229, 229, 0.14)
        );
}}

.brand-title {{

    font-size:
        22px;

    font-weight:
        850;

    color:
        #E8FCFF !important;

    letter-spacing:
        -0.4px;
}}

.brand-subtitle {{

    margin-top:
        5px;

    font-size:
        12px;

    color:
        #4FD2D5 !important;

    letter-spacing:
        2px;

    font-weight:
        650;
}}


/* =========================================================
   PREMIUM HERO
   ========================================================= */

.hero {{

    max-width:
        1120px;

    min-height:
        520px;

    margin:
        46px auto 30px;

    padding:
        64px 74px;

    border-radius:
        38px;

    position:
        relative;

    overflow:
        hidden;

    background:

        linear-gradient(
            90deg,

            rgba(3, 18, 29, 0.99) 0%,

            rgba(4, 21, 33, 0.96) 38%,

            rgba(4, 23, 36, 0.78) 62%,

            rgba(4, 23, 36, 0.42) 100%
        ),

        url("{HERO_IMAGE_URI}");

    background-size:
        cover,
        57% auto;

    background-position:
        center,
        right center;

    background-repeat:
        no-repeat;

    border:
        1px solid
        rgba(0, 229, 229, 0.25);

    box-shadow:

        0 30px 90px
        rgba(0, 0, 0, 0.48),

        inset 0 1px 0
        rgba(255,255,255,0.045);

    isolation:
        isolate;
}}


/* =========================================================
   MEDICAL TECH GRID
   ========================================================= */

.hero::before {{

    content:
        "";

    position:
        absolute;

    inset:
        0;

    background-image:

        linear-gradient(
            rgba(70, 225, 225, 0.045)
            1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(70, 225, 225, 0.045)
            1px,
            transparent 1px
        );

    background-size:
        42px 42px;

    mask-image:
        linear-gradient(
            to right,
            black 0%,
            rgba(0,0,0,0.8) 42%,
            transparent 100%
        );

    pointer-events:
        none;

    z-index:
        -1;
}}


/* =========================================================
   RETINA IMAGE GLOW
   ========================================================= */

.hero::after {{

    content:
        "";

    position:
        absolute;

    width:
        620px;

    height:
        620px;

    right:
        -170px;

    top:
        -50px;

    background:

        radial-gradient(
            circle,
            rgba(0, 235, 225, 0.17)
            0%,

            rgba(0, 180, 195, 0.09)
            32%,

            transparent
            72%
        );

    pointer-events:
        none;

    z-index:
        0;
}}


/* =========================================================
   HERO CONTENT
   ========================================================= */

.hero > * {{

    position:
        relative;

    z-index:
        2;
}}


/* =========================================================
   HERO BADGE
   ========================================================= */

.hero-badge {{

    display:
        inline-flex;

    align-items:
        center;

    gap:
        9px;

    padding:
        10px 19px;

    border-radius:
        999px;

    background:
        rgba(0, 229, 229, 0.085);

    border:
        1px solid
        rgba(0, 229, 229, 0.42);

    color:
        #72FFF2 !important;

    font-size:
        14px;

    font-weight:
        850;

    letter-spacing:
        0.55px;

    box-shadow:
        0 0 28px
        rgba(0, 229, 229, 0.08);
}}


/* =========================================================
   HERO TITLE
   ========================================================= */

.hero-title {{

    margin-top:
        38px;

    font-size:
        clamp(58px, 8vw, 92px);

    line-height:
        0.98;

    font-weight:
        900;

    letter-spacing:
        -5px;

    white-space:
        nowrap;
}}


/* Retina */

.hero-title .title-white {{

    color:
        #F2FCFF !important;

    text-shadow:

        0 3px 18px
        rgba(0,0,0,0.42),

        0 0 35px
        rgba(225,255,255,0.08);
}}


/* Vision */

.hero-title .title-cyan {{

    color:
        #20E0D4 !important;

    text-shadow:

        0 0 25px
        rgba(32,224,212,0.25);
}}


/* =========================================================
   HERO DESCRIPTION
   ========================================================= */

.hero-description {{

    max-width:
        760px;

    margin-top:
        30px;

    font-size:
        21px;

    line-height:
        1.75;

    font-weight:
        550;

    color:
        #C3DDE5 !important;

    text-shadow:
        0 2px 13px
        rgba(0,0,0,0.5);
}}


/* =========================================================
   DETECT / SEGMENT / ANALYZE
   ========================================================= */

.hero-pipeline {{

    display:
        flex;

    align-items:
        center;

    gap:
        16px;

    margin-top:
        28px;

    font-size:
        15px;

    font-weight:
        850;

    letter-spacing:
        1.3px;

    color:
        #E0FFFF !important;
}}

.hero-pipeline .pipeline-item {{

    color:
        #78FFF3 !important;

    text-shadow:
        0 0 17px
        rgba(114,255,242,0.20);
}}

.hero-pipeline .pipeline-separator {{

    color:
        #25E2D7 !important;

    font-size:
        19px;

    font-weight:
        900;

    text-shadow:
        0 0 13px
        rgba(34,225,213,0.38);
}}


/* =========================================================
   CREATOR
   ========================================================= */

.creator {{

    margin-top:
        34px;

    font-size:
        14px;

    line-height:
        1.7;

    color:
        #8BA9B5 !important;
}}

.creator strong {{

    display:
        block;

    margin-top:
        2px;

    font-size:
        16px;

    font-weight:
        800;

    color:
        #8DFFF4 !important;

    text-shadow:
        0 0 14px
        rgba(141,255,244,0.15);
}}


/* =========================================================
   SECTION TITLES
   ========================================================= */

.section-title {{

    max-width:
        1120px;

    margin:
        58px auto 18px;

    font-size:
        25px;

    font-weight:
        850;

    color:
        #E2FCFD !important;
}}


/* =========================================================
   GLASS CARDS
   ========================================================= */

.glass-card {{

    background:
        linear-gradient(
            145deg,
            rgba(9,31,45,0.88),
            rgba(4,20,31,0.88)
        ) !important;

    border:
        1px solid
        rgba(93,220,220,0.14) !important;

    border-radius:
        22px !important;

    box-shadow:
        0 18px 45px
        rgba(0,0,0,0.22) !important;
}}


/* =========================================================
   MODEL CARDS
   ========================================================= */

.model-card {{

    padding:
        25px;

    min-height:
        130px;

    background:

        linear-gradient(
            145deg,
            rgba(12,40,55,0.92),
            rgba(5,23,35,0.92)
        );

    border:
        1px solid
        rgba(47,220,211,0.16);

    border-radius:
        20px;

    box-shadow:
        inset 0 1px 0
        rgba(255,255,255,0.025);
}}

.model-label {{

    color:
        #76A5B1 !important;

    font-size:
        12px;

    font-weight:
        700;

    letter-spacing:
        1.4px;

    text-transform:
        uppercase;
}}

.model-value {{

    margin-top:
        10px;

    color:
        #E5FFFF !important;

    font-size:
        25px;

    font-weight:
        850;
}}

.model-accent {{

    color:
        #28E0D4 !important;
}}


/* =========================================================
   BUTTONS
   ========================================================= */

button {{

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease !important;
}}

button:hover {{

    transform:
        translateY(-2px);

    box-shadow:
        0 12px 28px
        rgba(0,229,229,0.15) !important;
}}

.primary-button {{

    background:
        linear-gradient(
            135deg,
            #19D8D0,
            #16B9D2
        ) !important;

    color:
        #021014 !important;

    border:
        none !important;

    font-weight:
        850 !important;

    border-radius:
        13px !important;
}}

.sample-button {{

    background:
        rgba(0,229,229,0.075) !important;

    color:
        #72F4EA !important;

    border:
        1px solid
        rgba(0,229,229,0.35) !important;

    border-radius:
        13px !important;

    font-weight:
        800 !important;
}}


/* =========================================================
   IMAGE BOXES
   ========================================================= */

.image-box {{

    border:
        1px solid
        rgba(71,215,216,0.13);

    border-radius:
        18px;

    overflow:
        hidden;
}}


/* =========================================================
   ANALYSIS SUCCESS
   ========================================================= */

.analysis-success {{

    margin-top:
        15px;

    padding:
        22px;

    border-radius:
        18px;

    background:
        linear-gradient(
            145deg,
            rgba(20,160,145,0.09),
            rgba(8,50,50,0.12)
        );

    border:
        1px solid
        rgba(50,224,203,0.20);
}}

.status-header {{

    display:
        flex;

    align-items:
        center;

    gap:
        12px;
}}

.status-dot {{

    width:
        9px;

    height:
        9px;

    border-radius:
        50%;

    background:
        #50F0D3;

    box-shadow:
        0 0 14px
        rgba(80,240,211,0.75);
}}

.status-title {{

    color:
        #55F2D9 !important;

    font-size:
        14px;

    font-weight:
        850;

    letter-spacing:
        1.5px;
}}

.status-subtitle {{

    margin-top:
        3px;

    color:
        #7799A4 !important;

    font-size:
        12px;
}}

.metric-grid {{

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap:
        12px;

    margin-top:
        18px;
}}

.metric {{

    padding:
        15px;

    border-radius:
        13px;

    background:
        rgba(255,255,255,0.025);
}}

.metric-label {{

    display:
        block;

    color:
        #71919D !important;

    font-size:
        11px;

    text-transform:
        uppercase;

    letter-spacing:
        1px;
}}

.metric-value {{

    display:
        block;

    margin-top:
        7px;

    color:
        #DFFFFB !important;

    font-size:
        20px;

    font-weight:
        800;
}}


/* =========================================================
   ERROR
   ========================================================= */

.analysis-error {{

    margin-top:
        15px;

    padding:
        18px;

    border-radius:
        14px;

    background:
        rgba(210,50,70,0.08);

    border:
        1px solid
        rgba(255,80,100,0.20);

    color:
        #FF9DA8 !important;
}}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {{

    max-width:
        1120px;

    margin:
        70px auto 30px;

    padding-top:
        25px;

    border-top:
        1px solid
        rgba(75,210,210,0.10);

    text-align:
        center;

    color:
        #5F7D88 !important;

    font-size:
        12px;
}}


/* =========================================================
   MOBILE
   ========================================================= */

@media(max-width: 800px) {{

    .hero {{

        margin:
            25px 12px;

        padding:
            45px 28px;

        min-height:
            500px;

        background-size:
            cover,
            90% auto;

        background-position:
            center,
            right bottom;
    }}

    .hero-title {{

        font-size:
            55px;

        letter-spacing:
            -3px;

        white-space:
            normal;
    }}

    .hero-description {{

        font-size:
            17px;
    }}

    .hero-pipeline {{

        gap:
            10px;

        font-size:
            13px;
    }}

    .metric-grid {{

        grid-template-columns:
            repeat(2, 1fr);
    }}

    .brand-wrapper {{

        padding-left:
            18px;

        padding-right:
            18px;
    }}
}}
"""


# ============================================================
# APPLICATION
# ============================================================

with gr.Blocks(
    title="Retina Vision AI"
) as demo:


    # ========================================================
    # BRAND HEADER
    # ========================================================

    gr.HTML(
        f"""
        <div class="brand-wrapper">

            <div class="brand">

                <div class="brand-logo">
                    {LOGO}
                </div>

                <div>

                    <div class="brand-title">
                        Retina Vision AI
                    </div>

                    <div class="brand-subtitle">
                        MEDICAL IMAGING INTELLIGENCE
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-badge">
                ✦ AI-POWERED MEDICAL IMAGING
            </div>


            <div class="hero-title">

                <span class="title-white">
                    Retina
                </span>

                <span class="title-cyan">
                    Vision
                </span>

            </div>


            <div class="hero-description">

                Intelligent retinal blood vessel
                segmentation for advanced medical
                image analysis.

            </div>


            <div class="hero-pipeline">

                <span class="pipeline-item">
                    DETECT
                </span>

                <span class="pipeline-separator">
                    |
                </span>

                <span class="pipeline-item">
                    SEGMENT
                </span>

                <span class="pipeline-separator">
                    |
                </span>

                <span class="pipeline-item">
                    ANALYZE
                </span>

            </div>


            <div class="creator">

                Designed & engineered by

                <strong>
                    Voona Manikantha
                </strong>

            </div>

        </div>
        """
    )


    # ========================================================
    # MODEL INTELLIGENCE
    # ========================================================

    gr.HTML(
        """
        <div class="section-title">
            Model Intelligence
        </div>
        """
    )


    with gr.Row():

        with gr.Column():

            gr.HTML(
                """
                <div class="model-card">

                    <div class="model-label">
                        Architecture
                    </div>

                    <div class="model-value">
                        U-Net
                    </div>

                    <div class="model-label">
                        Semantic Segmentation
                    </div>

                </div>
                """
            )


        with gr.Column():

            gr.HTML(
                """
                <div class="model-card">

                    <div class="model-label">
                        Encoder
                    </div>

                    <div class="model-value">

                        <span class="model-accent">
                            ResNet34
                        </span>

                    </div>

                    <div class="model-label">
                        ImageNet pretrained
                    </div>

                </div>
                """
            )


        with gr.Column():

            gr.HTML(
                """
                <div class="model-card">

                    <div class="model-label">
                        Benchmark Dice
                    </div>

                    <div class="model-value">

                        <span class="model-accent">
                            0.7113
                        </span>

                    </div>

                    <div class="model-label">
                        Test benchmark
                    </div>

                </div>
                """
            )


        with gr.Column():

            gr.HTML(
                """
                <div class="model-card">

                    <div class="model-label">
                        IoU
                    </div>

                    <div class="model-value">

                        <span class="model-accent">
                            0.5523
                        </span>

                    </div>

                    <div class="model-label">
                        Benchmark score
                    </div>

                </div>
                """
            )


    # ========================================================
    # LIVE ANALYSIS
    # ========================================================

    gr.HTML(
        """
        <div class="section-title">
            Live Retina Analysis
        </div>
        """
    )


    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes="glass-card"
        ):

            gr.Markdown(
                "### Retinal Image"
            )

            input_image = gr.Image(
                type="numpy",
                label="Upload retinal image"
            )


            with gr.Row():

                analyze_button = gr.Button(
                    "Analyze Retina",
                    variant="primary",
                    elem_classes="primary-button"
                )

                sample_button = gr.Button(
                    "Try Sample Retina",
                    elem_classes="sample-button"
                )


        with gr.Column(
            scale=1,
            elem_classes="glass-card"
        ):

            gr.Markdown(
                "### Vessel Segmentation"
            )

            output_mask = gr.Image(
                label="Segmentation mask"
            )


    # ========================================================
    # OUTPUT VISUALIZATIONS
    # ========================================================

    with gr.Row():

        with gr.Column(
            elem_classes="glass-card"
        ):

            gr.Markdown(
                "### Vessel Overlay"
            )

            output_overlay = gr.Image(
                label="Overlay"
            )


        with gr.Column(
            elem_classes="glass-card"
        ):

            gr.Markdown(
                "### Probability Map"
            )

            output_heatmap = gr.Image(
                label="Model probability"
            )


    status_output = gr.HTML()


    # ========================================================
    # BUTTON EVENTS
    # ========================================================

    analyze_button.click(
        fn=analyze_image,
        inputs=input_image,
        outputs=[
            input_image,
            output_mask,
            output_overlay,
            output_heatmap,
            status_output
        ]
    )


    sample_button.click(
        fn=try_sample,
        inputs=None,
        outputs=[
            input_image,
            output_mask,
            output_overlay,
            output_heatmap,
            status_output
        ]
    )


    # ========================================================
    # AI PROCESSING PIPELINE
    # ========================================================

    gr.HTML(
        """
        <div class="section-title">
            AI Processing Pipeline
        </div>


        <div class="model-card">

            <div style="
                display:grid;
                grid-template-columns:
                repeat(4,1fr);
                gap:20px;
            ">


                <div>

                    <div class="model-label">
                        01
                    </div>

                    <div class="model-value">
                        Input
                    </div>

                    <div class="model-label">
                        Fundus image
                    </div>

                </div>


                <div>

                    <div class="model-label">
                        02
                    </div>

                    <div class="model-value">
                        Encode
                    </div>

                    <div class="model-label">
                        ResNet34
                    </div>

                </div>


                <div>

                    <div class="model-label">
                        03
                    </div>

                    <div class="model-value">
                        Segment
                    </div>

                    <div class="model-label">
                        U-Net decoder
                    </div>

                </div>


                <div>

                    <div class="model-label">
                        04
                    </div>

                    <div class="model-value">
                        Analyze
                    </div>

                    <div class="model-label">
                        Vessel probability
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # TECHNICAL SPECIFICATIONS
    # ========================================================

    gr.HTML(
        """
        <div class="section-title">
            Technical Specifications
        </div>


        <div class="model-card">

            <div style="
                display:grid;
                grid-template-columns:
                repeat(2,1fr);
                gap:22px;
            ">


                <div>

                    <div class="model-label">
                        Framework
                    </div>

                    <div class="model-value">
                        PyTorch
                    </div>

                </div>


                <div>

                    <div class="model-label">
                        Model
                    </div>

                    <div class="model-value">
                        U-Net + ResNet34
                    </div>

                </div>


                <div>

                    <div class="model-label">
                        Loss Function
                    </div>

                    <div class="model-value">
                        Tversky + BCE
                    </div>

                </div>


                <div>

                    <div class="model-label">
                        Decision Threshold
                    </div>

                    <div class="model-value">
                        0.60
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">

            Retina Vision AI · Deep Learning
            · Medical Image Segmentation

            <br><br>

            Built by Voona Manikantha

        </div>
        """
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7862))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CSS,
        theme=gr.themes.Base(),
        show_error=True
    )