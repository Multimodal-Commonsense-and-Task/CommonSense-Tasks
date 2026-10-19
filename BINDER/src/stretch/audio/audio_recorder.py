#!/usr/bin/env python3

# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

"""
A simple audio recorder using PyAudio.
"""

import audioop
import wave
from typing import List

import numpy as np
import pyaudio
from tqdm import tqdm

from stretch.audio.base import AbstractSpeechToText


class AudioRecorder:
    """
    A class for recording audio from a microphone OR from a fake WAV file (debug mode).
    """

    def __init__(
        self,
        chunk: int = 1024,
        channels: int = 2,
        sample_rate: int = 44100,
        volume_threshold: int = 500,
        debug: bool = False,
        debug_wav_path: str = "./dummy.wav",
    ) -> None:
        """
        Initialize the AudioRecorder.

        Args:
            chunk (int): Number of frames per buffer. Defaults to 1024.
            channels (int): Number of channels to record. Defaults to 2.
            sample_rate (int): Sampling rate. Defaults to 44100 Hz.
            volume_threshold (int): Volume threshold.
            debug (bool): If True, load audio from debug_wav_path instead of mic.
            debug_wav_path (str): Path to the dummy wav file.
        """
        self.chunk = chunk
        self.channels = channels
        self.sample_rate = sample_rate
        self.format = pyaudio.paInt16
        self.volume_threshold = volume_threshold
        self.debug = debug
        self.debug_wav_path = debug_wav_path

        self.frames = []

        if not self.debug:
            self.audio = pyaudio.PyAudio()
            self.stream = None
            self.start()
        else:
            print(f"[DEBUG] AudioRecorder running in debug mode. "
                  f"Will load WAV: {self.debug_wav_path}")

    def reset(self):
        self.frames = []

    def start(self):
        if self.debug:
            return
        self.stream = self.get_stream()
        self.stream.stop_stream()

    def record(self, filename="recording.wav", duration=10.0, silence_limit=1.0):
        """
        Record audio normally OR load dummy WAV when in debug mode.
        """
        if self.debug:
            print("[DEBUG] Using dummy WAV")
            import wave

            with wave.open(self.debug_wav_path, "rb") as src:
                params = src.getparams()
                frames = src.readframes(src.getnframes())

            with wave.open(filename, "wb") as dst:
                dst.setparams(params)
                dst.writeframes(frames)

            print(f"[DEBUG] Copied dummy wav to: {filename}")
            return


        # Mic recording mode
        audio_started = False
        silent_chunks = 0

        print("Recording...")
        if self.stream is None:
            self.start()

        self.stream.start_stream()

        for _ in tqdm(range(0, int(self.sample_rate / self.chunk * duration))):
            data = self.stream.read(self.chunk)
            self.frames.append(data)

            rms = audioop.rms(data, 2)

            if audio_started:
                self.frames.append(data)

            if rms > self.volume_threshold:
                audio_started = True
                silent_chunks = 0
            elif audio_started:
                silent_chunks += 1

            if silent_chunks > silence_limit * self.sample_rate / self.chunk:
                break

        print("Recording finished.")
        self.stream.stop_stream()
        self.save(filename)
        self.reset()

    def __del__(self):
        if not self.debug:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            if hasattr(self, "audio"):
                self.audio.terminate()

    def get_stream(self):
        return self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def save(self, filename):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self.frames))
        print(f"Audio saved as {filename}")


if __name__ == "__main__":
    recorder = AudioRecorder()
    # Records 5 seconds of audio and saves it as "recording.wav"
    recorder.record("recording.wav", duration=5)

