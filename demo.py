import time

import gradio as gr
import torch
import torchaudio

from vui.inference import asr, render
from vui.model import Vui


def get_available_models():
    """Extract all CAPs static variables from Vui class that end with .pt"""
    models = {}
    for attr_name in dir(Vui):
        if attr_name.isupper():
            attr_value = getattr(Vui, attr_name)
            if isinstance(attr_value, str) and attr_value.endswith(".pt"):
                models[attr_name] = attr_value
    return models


AVAILABLE_MODELS = get_available_models()
print(f"Available models: {list(AVAILABLE_MODELS.keys())}")

current_model = None
current_model_name = None


def load_and_warm_model(model_name):
    """Load and warm up a specific model"""
    global current_model, current_model_name

    if current_model_name == model_name and current_model is not None:
        print(f"Model {model_name} already loaded and warmed up!")
        return current_model

    print(f"Loading model {model_name}...")
    model_path = AVAILABLE_MODELS[model_name]
    model = Vui.from_pretrained_inf(model_path).cuda()

    print(f"Compiling model {model_name}...")
    model.decoder = torch.compile(model.decoder, fullgraph=True)

    print(f"Warming up model {model_name}...")
    warmup_text = "Hello, this is a test. Let's say some random shizz"
    render(
        model,
        warmup_text,
        max_secs=5,
    )

    current_model = model
    current_model_name = model_name
    print(f"Model {model_name} loaded and warmed up successfully!")
    return model


# Load default model (COHOST)
default_model = "BASE"
default_model = (
    "BASE" if "BASE" in AVAILABLE_MODELS else list(AVAILABLE_MODELS.keys())[0]
)
model = load_and_warm_model(default_model)

# Preload sample 1 (index 0) with current model
print("Preloading sample 1...")
sample_1_text = """Welcome to Fluxions, the podcast where... we uh explore how technology is shaping the world around us. I'm your host, Alex.
[breath] And I'm Jamie um [laugh] today, we're diving into a [hesitate] topic that's transforming customer service uh voice technology for agents.
That's right. We're [hesitate] talking about the AI-driven tools that are making those long, frustrating customer service calls a little more bearable, for both the customer and the agents."""

sample_1_audio = render(
    current_model,
    sample_1_text,
)
sample_1_audio = sample_1_audio.cpu()
sample_1_audio = sample_1_audio[..., :-2000]  # Trim end artifacts
preloaded_sample_1 = (model.codec.config.sample_rate, sample_1_audio.flatten().numpy())
print("Sample 1 preloaded successfully!")
print("Models ready for inference!")

# Sample texts for quick testing - keeping original examples intact
SAMPLE_TEXTS = [
    """Welcome to Fluxions, the podcast where... we uh explore how technology is shaping the world around us. I'm your host, Alex.
[breath] And I'm Jamie um [laugh] today, we're diving into a [hesitate] topic that's transforming customer service uh voice technology for agents.
That's right. We're [hesitate] talking about the AI-driven tools that are making those long, frustrating customer service calls a little more bearable, for both the customer and the agents.""",
    """Um, hey Sarah, so I just left the meeting with the, uh, rabbit focus group and they are absolutely loving the new heritage carrots! Like, I've never seen such enthusiastic thumping in my life! The purple ones are testing through the roof - apparently the flavor profile is just amazing - and they're willing to pay a premium for them! We need to, like, triple production on those immediately and maybe consider a subscription model? Anyway, gotta go, but let's touch base tomorrow about scaling this before the Easter rush hits!""",
    """What an absolute joke, like I'm really not enjoying this situation where I'm just forced to say things.""",
    """ So [breath] I don't know if you've been there [breath] but I'm really pissed off.
Oh no! Why, what happened?
Well I went to this cafe hearth, and they gave me the worst toastie I've ever had, it didn't come with salad it was just raw.
Well that's awful what kind of toastie was it?
It was supposed to be a chicken bacon lettuce tomatoe, but it was fucking shite, like really bad and I honestly would have preferred to eat my own shit.
[laugh] well, it must have been awful for you, I'm sorry to hear that, why don't we move on to brighter topics, like the good old weather?""",
]


def text_to_speech(
    text, prompt_audio=None, temperature=0.5, top_k=100, top_p=None, max_duration=120
):
    """
    Convert text to speech using the current Vui model

    Args:
        text (str): Input text to convert to speech
        prompt_audio (tuple): Optional (sample_rate, audio_array) from Gradio audio input
        temperature (float): Sampling temperature (0.1-1.0)
        top_k (int): Top-k sampling parameter
        top_p (float): Top-p sampling parameter (None to disable)
        max_duration (int): Maximum audio duration in seconds

    Returns:
        tuple: (sample_rate, audio_array) for Gradio audio output
    """
    if not text.strip():
        return None, "Please enter some text to convert to speech."

    if current_model is None:
        return None, "No model loaded. Please select a model first."

    print(f"Generating speech for: {text[:50]}... using model {current_model_name}")

    # Process prompt audio if provided
    prompt_codes = None
    prompt_text = ""
    if prompt_audio is not None:
        sr, audio = prompt_audio

        audio = torch.from_numpy(audio).float()
        audio = audio / audio.abs().max()
        if len(audio.shape) > 1:
            audio = audio.mean(1)

        codec_sr = current_model.codec.config.sample_rate
        # Limit to 30 seconds
        max_samples = int(30 * codec_sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        torchaudio.save("prompt_audio.wav", audio[None], sr)
        print(audio.shape)

        # Resample to codec sample rate if needed
        if sr != codec_sr:
            resampler = torchaudio.transforms.Resample(sr, codec_sr)
            audio = resampler(audio)

        # Encode audio to get prompt codes
        with torch.inference_mode():
            audio = audio[None, None]
            prompt_codes = current_model.codec.encode(audio.cuda())

        resampler = torchaudio.transforms.Resample(codec_sr, 16000)
        prompt_text = asr(resampler(audio.flatten()))
        print("PROMPT_TEXT", prompt_text)

        print(f"Using audio prompt with shape: {prompt_codes.shape}")

    # Generate speech using render
    t1 = time.perf_counter()
    print(prompt_text + text)
    result = render(
        current_model,
        (prompt_text + " " + text).strip(),
        prompt_codes=prompt_codes,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_secs=max_duration,
    )

    # Long text: render returns (codes, text, audio) tuple
    waveform = result

    # waveform is already decoded audio from generate_infinite
    waveform = waveform.cpu()
    sr = current_model.codec.config.sample_rate

    # Calculate generation speed
    generation_time = time.perf_counter() - t1
    audio_duration = waveform.shape[-1] / sr
    speed_factor = audio_duration / generation_time

    # Trim end artifacts if needed
    if waveform.shape[-1] > 2000:
        waveform = waveform[..., :-2000]

    # Convert to numpy array for Gradio
    audio_array = waveform.flatten().numpy()

    info = f"Generated {audio_duration:.1f}s of audio in {generation_time:.1f}s ({speed_factor:.1f}x realtime) with {current_model_name}"
    print(info)

    return (sr, audio_array), info


def change_model(model_name):
    """Change the active model and return status"""
    try:
        load_and_warm_model(model_name)
        return f"Successfully loaded and warmed up model: {model_name}"
    except Exception as e:
        return f"Error loading model {model_name}: {str(e)}"


def load_sample_text(sample_index):
    """Load a sample text for quick testing"""
    if 0 <= sample_index < len(SAMPLE_TEXTS):
        return SAMPLE_TEXTS[sample_index]
    return ""


# Create Gradio interface
with gr.Blocks(
    title="Vui",
    theme=gr.themes.Soft(),
    head="""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to generate (but not when Shift is pressed)
        if ((e.ctrlKey) && e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const generateBtn = document.querySelector('button[variant="primary"]');
            if (generateBtn && !generateBtn.disabled) {
                generateBtn.click();
            }
        }
        else if ((e.ctrlKey) && e.code === 'Space') {
            e.preventDefault();
            const audioElement = document.querySelector('audio');
            if (audioElement) {
                if (audioElement.paused) {
                    audioElement.play();
                } else {
                    audioElement.pause();
                }
            }
        }
    });

    // Auto-play audio when it's updated
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const audioElements = document.querySelectorAll('audio');
                audioElements.forEach(function(audio) {
                    if (audio.src && !audio.dataset.hasAutoplayListener) {
                        audio.dataset.hasAutoplayListener = 'true';
                        audio.addEventListener('loadeddata', function() {
                            // Small delay to ensure audio is ready
                            setTimeout(() => {
                                audio.play().catch(e => {
                                    console.log('Autoplay prevented by browser:', e);
                                });
                            }, 100);
                        });
                    }
                });
            }
        });
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

});
</script>
""",
) as demo:

    gr.Markdown(
        "**Keyboard Shortcuts:** `Ctrl + Enter` to generate` or Ctrl + Space to pause"
    )

    with gr.Row():
        with gr.Column(scale=2):
            # Model selector
            model_dropdown = gr.Dropdown(
                choices=list(AVAILABLE_MODELS.keys()),
                value=default_model,
                label=None,
                info="Select a voice model",
            )

            # Model status
            model_status = gr.Textbox(
                label=None,
                value=f"Model {default_model} loaded and ready",
                interactive=False,
                lines=1,
            )

            # Audio input for voice prompt
            audio_input = gr.Audio(
                label="Voice Prompt (optional) - Upload up to 30s of audio to use as voice style prompt",
                type="numpy",
                format="wav",
                waveform_options={"sample_rate": 22050},
            )

            # Text input
            text_input = gr.Textbox(
                label=None,
                placeholder="Enter the text you want to convert to speech...",
                lines=5,
                max_lines=10,
            )

        with gr.Column(scale=1):
            # Audio output with autoplay
            audio_output = gr.Audio(
                label="Generated Speech", type="numpy", autoplay=True  # Enable autoplay
            )

            # Info output
            info_output = gr.Textbox(
                label="Generation Info", lines=3, interactive=False
            )

    with gr.Row():
        with gr.Column(scale=2):

            # Sample text buttons
            gr.Markdown("**Quick samples:**")
            with gr.Row():
                sample_btns = []
                for i, sample in enumerate(SAMPLE_TEXTS):
                    btn = gr.Button(f"Sample {i+1}", size="sm")
                    if i == 0:  # Sample 1 (index 0) - use preloaded audio

                        def load_preloaded_sample_1():
                            return (
                                SAMPLE_TEXTS[0],
                                preloaded_sample_1,
                                "Preloaded sample 1 audio",
                            )

                        btn.click(
                            fn=load_preloaded_sample_1,
                            outputs=[text_input, audio_output, info_output],
                        )
                    else:
                        btn.click(
                            fn=lambda idx=i: SAMPLE_TEXTS[idx], outputs=text_input
                        )

            # Generation parameters
            with gr.Accordion("Advanced Settings", open=False):
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label="Temperature",
                    info="Higher values = more varied speech",
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=200,
                    value=100,
                    step=1,
                    label="Top-K",
                    info="Number of top tokens to consider",
                )

                use_top_p = gr.Checkbox(label="Use Top-P sampling", value=False)
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top-P",
                    info="Cumulative probability threshold",
                    visible=False,
                )

                max_duration = gr.Slider(
                    minimum=5,
                    maximum=120,
                    value=120,
                    step=5,
                    label="Max Duration (seconds)",
                    info="Maximum length of generated audio",
                )

                # Show/hide top_p based on checkbox
                use_top_p.change(
                    fn=lambda x: gr.update(visible=x), inputs=use_top_p, outputs=top_p
                )

            # Generate button
            generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

    # Examples section
    gr.Markdown("## 📝 Example Texts")
    with gr.Accordion("View example texts", open=False):
        for i, sample in enumerate(SAMPLE_TEXTS):
            gr.Markdown(f"**Sample {i+1}:** {sample}")

    # Connect the model change function
    model_dropdown.change(fn=change_model, inputs=model_dropdown, outputs=model_status)

    # Connect the generate function
    def generate_wrapper(text, prompt_audio, temp, k, use_p, p, duration):
        top_p_val = p if use_p else None

        # If audio prompt is provided, switch to BASE model
        if prompt_audio is not None:
            if current_model_name != "BASE":
                change_model("BASE")

        return text_to_speech(text, prompt_audio, temp, k, top_p_val, duration)

    generate_btn.click(
        fn=generate_wrapper,
        inputs=[
            text_input,
            audio_input,
            temperature,
            top_k,
            use_top_p,
            top_p,
            max_duration,
        ],
        outputs=[audio_output, info_output],
    )

    # Also allow Enter key to generate
    text_input.submit(
        fn=generate_wrapper,
        inputs=[
            text_input,
            audio_input,
            temperature,
            top_k,
            use_top_p,
            top_p,
            max_duration,
        ],
        outputs=[audio_output, info_output],
    )

    # Auto-load sample 1 on startup
    demo.load(
        fn=lambda: (
            SAMPLE_TEXTS[0],
            preloaded_sample_1,
            "Sample 1 preloaded and ready!",
        ),
        outputs=[text_input, audio_output, info_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
