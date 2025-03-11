import os 
import logging

import soundfile as sf
import librosa
import ctranslate2
import transformers
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

from src.utils.memory import GPUMemoryManager
from src.telemetry.prometheus import telemetry

logger = logging.getLogger(__name__)

class WhisperTranscribe:
    def __init__(self, cfg: DictConfig):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sampling_rate = 16000
        self.mem_manager = GPUMemoryManager()
        self.cfg = cfg
        
        self._init_processor_and_model()
        
        self._warmup()
    
    def _init_processor_and_model(self):
        try:
            model_name = self.cfg.model_name
            logger.info(f"Loading Whisper processor: {model_name}")
            self.processor = transformers.WhisperProcessor.from_pretrained(model_name)
        except Exception as e:
            logger.error(f'Error loading Whisper processor: {e}')
            raise
            
        try:
            ct2_path = self.cfg.paths.ct2_model
            if not os.path.exists(ct2_path):
                logger.warning(f"CTranslate2 model not found at '{ct2_path}'")
                raise FileNotFoundError(f'CTranslate2 model not found at "{ct2_path}"')
            
            compute_type = self.cfg.performance.compute_type if hasattr(self.cfg, 'performance') else 'float16'
            logger.info(f"Loading CTranslate2 model from {ct2_path} with compute_type={compute_type}")
            
            self.model = ctranslate2.models.Whisper(
                ct2_path,
                device=self.device,
                compute_type=compute_type
            )
        except Exception as e:
            logger.error(f"Error loading CTranslate2 Whisper model: {e}")
            raise
    
    def _warmup(self):
        """Run model inference on dummy input"""
        try:
            logger.info("Warming up model with dummy input...")
            dummy_input = np.zeros((16000,), dtype=np.float32)  
            _ = self.generate(dummy_input)
            logger.info("Model warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    def _segment_audio(self, audio_data, segment_len=30):
        """Segments audio into chunks of specified length in seconds"""
        samples_per_segment = self.sampling_rate * segment_len
        num_segments = int(np.ceil(len(audio_data) / samples_per_segment))
        segments = np.array_split(audio_data, num_segments)
        return segments

    @telemetry.track_vram('Whisper')
    @telemetry.track_latency(telemetry.stt_latency)
    def generate(self, audio):
        """Generate transcription from audio input.
        
        Args:
            audio: Audio data as string path, numpy array, or tensor
            
        Returns:
            Transcribed text as string
        """
        # Load and normalize audio
        try:
            if isinstance(audio, str):
                logger.debug(f"Loading audio from file: {audio}")
                y, sr = librosa.load(audio, sr=self.sampling_rate, mono=True)
            elif isinstance(audio, np.ndarray):
                y = audio
                # Convert to mono if needed
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = y.mean(axis=1)
            else:
                raise TypeError(f"Unsupported audio type: {type(audio)}")
        except Exception as e:
            error_msg = f"Error loading audio: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Process audio in segments 
        segments = self._segment_audio(y)
        logger.debug(f"Split audio into {len(segments)} segments")
        
        transcription = ""
        for i, segment in enumerate(segments):
            try:
                inputs = self.processor(segment, return_tensors='pt', sampling_rate=self.sampling_rate)
                features = inputs.input_features
                
                # Utilize memory pool for GPU tensors if available
                if self.device == "cuda":
                    feature_key = f"whisper_features_{len(segment)}"
                    buffer_size = features.shape
                    pooled_features = self.mem_manager.allocate_buffer(
                        feature_key, buffer_size, features.dtype
                    )
                    pooled_features.copy_(features)
                    features = pooled_features
                
                features = features.to(self.device)
                storage_view = ctranslate2.StorageView.from_array(features)
                
                # Detect language
                if i == 0:
                    results_lang = self.model.detect_language(storage_view)
                    self.detected_lang, prob = results_lang[0][0]
                    logger.debug(f"Detected language: {self.detected_lang} (probability: {prob:.2f})")
                
                prompt = self.processor.tokenizer.convert_tokens_to_ids([
                    "<|startoftranscript|>",
                    self.detected_lang,
                    "<|transcribe|>",
                    "<|notimestamps|>",
                ])
                
                results = self.model.generate(storage_view, [prompt])
                segment_text = self.processor.decode(results[0].sequences_ids[0])
                transcription += segment_text.strip() + " "
                
            except Exception as e:
                error_msg = f"Error processing segment {i}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        return transcription.strip()


@hydra.main(config_path='../../conf/stt/', config_name='whisper', version_base='1.3')
def test_transcription(cfg: DictConfig):
    """Test the transcription model with sample audio."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        transcriber = WhisperTranscribe(cfg)
        
        audio_filepath = cfg.paths.get('test_audio', 'data/audio.mp3')
        
        if not os.path.exists(audio_filepath):
            logger.info(f"Test file not found, creating dummy audio at '{audio_filepath}'")
            os.makedirs(os.path.dirname(audio_filepath), exist_ok=True)
            dummy_audio = np.random.rand(3 * 16000).astype(np.float32) * 0.1
            sf.write(audio_filepath, dummy_audio, 16000)
            
        logger.info(f"Transcribing: {audio_filepath}")
        mem_stats = GPUMemoryManager().get_stats()
        logger.info(f'GPU memory stats: {mem_stats}')
        result = transcriber.generate(audio_filepath)
        logger.info(f"Transcription: {result}")
        
    except Exception as e:
        logger.error(f"Transcription test failed: {e}")
        raise


if __name__ == '__main__':
    test_transcription()