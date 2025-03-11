import logging
import hydra
from omegaconf import DictConfig

import torch
from TTS.api import TTS

from src.telemetry.prometheus import telemetry

logger = logging.getLogger(__name__)

class CoquiTTS:
    def __init__(self, cfg: DictConfig):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self._initialize_tts()
        
    def _initialize_tts(self):
        try:
            self.tts = TTS(self.cfg.model_name).to(self.device)
            self._verify_capabilities()
            if self.cfg.optimize_for_realtime:
                self.optimize_for_realtime()
        except Exception as e:
            logger.error(f"TTS initialization failed: {str(e)}")
            raise

    def _verify_capabilities(self):
        try:
            self.speakers = self.tts.speakers if hasattr(self.tts, 'speakers') else []
            self.languages = self.tts.languages if hasattr(self.tts, 'languages') else ['en']
            # logger.info(f"Model supports {len(self.speakers)} speakers: {self.speakers}")
        except Exception as e:
            logger.error(f"Capability detection failed: {str(e)}")
            self.speakers = []
            self.languages = ['en']

    @telemetry.track_vram('Coqui-TTS')
    @telemetry.track_latency(telemetry.tts_latency)
    def synthesize(self, text: str, speaker: str, output_path: str = None) -> str:
        output_path = output_path or self.cfg.default_output_path
        speaker = speaker or self.cfg.default_speaker
        
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker=speaker,
                # language=self.cfg.default_language
            )
            return output_path
        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            return None

    def optimize_for_realtime(self):
        if self.device == "cuda":
            try:
                self.tts = torch.compile(self.tts, 
                    mode='max-autotune')
                logger.info("TTS model compiled for realtime inference")
            except Exception as e:
                logger.warning(f"Realtime optimization failed: {str(e)}")

@hydra.main(config_path="../../conf/tts", config_name="coqui", version_base="1.3")
def test_tts(cfg: DictConfig):

    try:
        tts = CoquiTTS(cfg)
        test_text = "Hi, how are you!?, nice to meet you."
        
        output = tts.synthesize(test_text)
        print(f"Generated audio at: {output}")
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"TTS test failed: {str(e)}")

if __name__ == '__main__':
    test_tts()