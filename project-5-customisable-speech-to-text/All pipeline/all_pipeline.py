import logging
import os
import sys
from copy import copy
from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional
from sys import platform
from VAD.vad_handler import VADHandler
from arguments_classes.module_arguments import ModuleArguments
from arguments_classes.socket_receiver_arguments import SocketReceiverArguments
from arguments_classes.socket_sender_arguments import SocketSenderArguments
from arguments_classes.vad_arguments import VADHandlerArguments
from arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments
from arguments_classes.paraformer_stt_arguments import ParaformerSTTHandlerArguments
import torch
from rich.console import Console
from transformers import HfArgumentParser
from utils.thread_manager import ThreadManager

# Caching allows ~50% compilation time reduction
CURRENT_DIR = Path(__file__).resolve().parent
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(CURRENT_DIR, "tmp")

console = Console()
logging.getLogger("numba").setLevel(logging.WARNING)  # Quiet down numba logs

def rename_args(args, prefix):
    gen_kwargs = {}
    for key in copy(args.__dict__):
        if key.startswith(prefix):
            value = args.__dict__.pop(key)
            new_key = key[len(prefix) + 1:]  # Remove prefix and underscore
            if new_key.startswith("gen_"):
                gen_kwargs[new_key[4:]] = value  # Remove 'gen_' and add to dict
            else:
                args.__dict__[new_key] = value

    args.__dict__["gen_kwargs"] = gen_kwargs

def parse_arguments():
    parser = HfArgumentParser(
        (
            ModuleArguments,
            SocketReceiverArguments,
            SocketSenderArguments,
            VADHandlerArguments,
            WhisperSTTHandlerArguments,
            ParaformerSTTHandlerArguments,
        )
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        return parser.parse_args_into_dataclasses()

def setup_logger(log_level):
    global logger
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if log_level == "debug":
        torch._logging.set_logs(graph_breaks=True, recompiles=True, cudagraphs=True)

def optimal_mac_settings(mac_optimal_settings: Optional[str], *handler_kwargs):
    if mac_optimal_settings:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "device"):
                kwargs.device = "mps"
            if hasattr(kwargs, "mode"):
                kwargs.mode = "local"
            if hasattr(kwargs, "stt"):
                kwargs.stt = "whisper-mlx"

def check_mac_settings(module_kwargs):
    if platform == "darwin":
        if module_kwargs.device == "cuda":
            raise ValueError(
                "Cannot use CUDA on macOS. Please set the device to 'cpu' or 'mps'."
            )

def overwrite_device_argument(common_device: Optional[str], *handler_kwargs):
    if common_device:
        for kwargs in handler_kwargs:
            if hasattr(kwargs, "stt_device"):
                kwargs.stt_device = common_device

def prepare_module_args(module_kwargs, *handler_kwargs):
    optimal_mac_settings(module_kwargs.local_mac_optimal_settings, module_kwargs)
    if platform == "darwin":
        check_mac_settings(module_kwargs)
    overwrite_device_argument(module_kwargs.device, *handler_kwargs)

def prepare_all_args(
    module_kwargs,
    whisper_stt_handler_kwargs,
    paraformer_stt_handler_kwargs,
):
    prepare_module_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
    )

    rename_args(whisper_stt_handler_kwargs, "stt")
    rename_args(paraformer_stt_handler_kwargs, "paraformer_stt")

def initialize_queues_and_events():
    return {
        "stop_event": Event(),
        "should_listen": Event(),
        "recv_audio_chunks_queue": Queue(),
        "spoken_prompt_queue": Queue(),
        "text_prompt_queue": Queue(),
    }

def build_pipeline(
    module_kwargs,
    socket_receiver_kwargs,
    socket_sender_kwargs,
    vad_handler_kwargs,
    whisper_stt_handler_kwargs,
    paraformer_stt_handler_kwargs,
    queues_and_events,
):
    stop_event = queues_and_events["stop_event"]
    should_listen = queues_and_events["should_listen"]
    recv_audio_chunks_queue = queues_and_events["recv_audio_chunks_queue"]
    spoken_prompt_queue = queues_and_events["spoken_prompt_queue"]
    text_prompt_queue = queues_and_events["text_prompt_queue"]

    from connections.local_audio_streamer import LocalAudioStreamer

    local_audio_streamer = LocalAudioStreamer(
        input_queue=recv_audio_chunks_queue, output_queue=spoken_prompt_queue
    )
    comms_handlers = [local_audio_streamer]
    should_listen.set()

    vad = VADHandler(
        stop_event,
        queue_in=recv_audio_chunks_queue,
        queue_out=spoken_prompt_queue,
        setup_args=(should_listen,),
        setup_kwargs=vars(vad_handler_kwargs),
    )

    stt = get_stt_handler(
        module_kwargs,
        stop_event,
        spoken_prompt_queue,
        text_prompt_queue,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
    )

    return ThreadManager([*comms_handlers, vad, stt])

def get_stt_handler(
    module_kwargs,
    stop_event,
    spoken_prompt_queue,
    text_prompt_queue,
    whisper_stt_handler_kwargs,
    paraformer_stt_handler_kwargs,
):
    if module_kwargs.stt == "whisper":
        from STT.whisper_stt_handler import WhisperSTTHandler
        return WhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "whisper-mlx":
        from STT.lightning_whisper_mlx_handler import LightningWhisperSTTHandler
        return LightningWhisperSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(whisper_stt_handler_kwargs),
        )
    elif module_kwargs.stt == "paraformer":
        from STT.paraformer_handler import ParaformerSTTHandler
        return ParaformerSTTHandler(
            stop_event,
            queue_in=spoken_prompt_queue,
            queue_out=text_prompt_queue,
            setup_kwargs=vars(paraformer_stt_handler_kwargs),
        )
    else:
        raise ValueError("The STT should be either whisper, whisper-mlx, or paraformer.")

def main():
    (
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
    ) = parse_arguments()

    setup_logger(module_kwargs.log_level)

    prepare_all_args(
        module_kwargs,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
    )

    queues_and_events = initialize_queues_and_events()

    pipeline_manager = build_pipeline(
        module_kwargs,
        socket_receiver_kwargs,
        socket_sender_kwargs,
        vad_handler_kwargs,
        whisper_stt_handler_kwargs,
        paraformer_stt_handler_kwargs,
        queues_and_events,
    )

    try:
        pipeline_manager.start()
    except KeyboardInterrupt:
        pipeline_manager.stop()

if __name__ == "__main__":
    main()
