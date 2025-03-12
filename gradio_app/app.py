import os
import sys
import logging
import gradio as gr
import torch
import hydra
from omegaconf import DictConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stt.whisper_stt import WhisperTranscribe
from src.tts.coqui_tts import CoquiTTS
from src.llm.llama import SciLlama
from src.telemetry.prometheus import telemetry

logger = logging.getLogger(__name__)

class STEMTutorApp:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        try:
            self.transcriber = WhisperTranscribe(self.cfg.stt)
            self.tts = CoquiTTS(self.cfg.tts)
            self.llm = SciLlama(self.cfg.llm)
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            raise

    def _build_system_info(self):
        """System information"""
        info = []
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.append(f"GPU: {props.name} | VRAM: {props.total_memory/1024/1024/1024:.2f}GB |")
        info.append(f"STT Model: {self.cfg.stt.model_name} |")
        info.append(f"LLM Model: {self.cfg.llm.model.model_name_or_lora_path.split('/')[1]} |")
        info.append(f"TTS Model: {self.cfg.tts.model_name} |")
        return "\n".join(info)

    def process_query(self, audio_path: str, text_input: str, temperature: float, speaker: str) -> tuple:
        """Process either audio or text input"""
        try:
            torch.cuda.empty_cache()
            
            if text_input.strip():
                transcription = text_input
            else:
                if not audio_path:
                    return "", None, "No input provided"
                transcription = self.transcriber.generate(audio_path)
                if not transcription:
                    return "", None, "Transcription failed"
                
            # LLM processing
            llm_response = self.llm.generate(
                query=transcription,
                temperature=temperature,
            )
            
            # TTS synthesis
            audio_output = self.tts.synthesize(text=llm_response,
                                               speaker=speaker)
            
            return transcription, llm_response, audio_output
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return "", None, str(e)
        finally:
            torch.cuda.empty_cache()
        
    def create_interface(self):
        """Build interface"""
        with gr.Blocks(theme=gr.themes.Soft(), title="STEM Tutor") as demo:
            gr.Markdown("# STEM Tutor - Interactive Learning Assistant 🏫")
            gr.Markdown(self._build_system_info())
            
            with gr.Tabs():
                with gr.Tab("Ask Question"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Row():
                                audio_input = gr.Audio(
                                    sources=["microphone", "upload"],
                                    type="filepath",
                                    label="Ask via Audio"
                                )
                                text_input = gr.Textbox(
                                    label="Or Type Your Question",
                                    lines=3,
                                    placeholder="Enter your STEM question here..."
                                )
                            
                            process_btn = gr.Button("Process Question", variant="primary")
                        
                        with gr.Column():
                            transcription_out = gr.Textbox(
                                label="Your Question",
                                interactive=False,
                                visible=False
                            )

                            response_out = gr.Textbox(
                                label="AI Response",
                                elem_id="response-box",
                                lines=6,
                                max_lines=12
                            )
                            audio_output = gr.Audio(
                                label="Spoken Response",
                                autoplay=True,
                                visible=self.cfg.tts.enable_playback
                            )

                with gr.Tab("Settings"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### LLM Settings")
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature",
                                interactive=True
                            )
                        
                        with gr.Column():
                            gr.Markdown("### TTS Settings")
                            speaker = gr.Dropdown(
                                choices=self.tts.speakers,
                                value=self.cfg.tts.default_speaker,
                                label="Select Speaker voice",
                                interactive=True
                            )

                        process_btn.click(
                        fn=self.process_query,
                        inputs=[audio_input, text_input, temperature, speaker],
                        outputs=[transcription_out, response_out, audio_output],
                        show_progress="full"
                    )

            return demo

@hydra.main(config_path="../conf", config_name="app", version_base="1.3")
def launch_app(cfg: DictConfig):
    telemetry_cfg = cfg.telemetry
    telemetry.start_server(port=telemetry_cfg.port)
    
    logging.basicConfig(
        level=cfg.logging.level,
        format=cfg.logging.format
    )
    
    try:
        app = STEMTutorApp(cfg)
        demo = app.create_interface()
        demo.launch(
            server_name=cfg.app.host,
            server_port=cfg.app.port,
            share=cfg.app.share,
        )
    except Exception as e:
        logger.error(f"Application failed to start: {str(e)}")
        raise

if __name__ == "__main__":
    launch_app()