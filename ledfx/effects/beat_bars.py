# fork addition: Beat Bars — N evenly-spaced "pillars" along the strip,
# each pulsing in unison on the beat with an attack-release envelope.
# Sharp attack at the downbeat of each beat; tunable decay tail.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


class BeatBarsAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Beat Bars"
    CATEGORY = "BPM"
    HIDDEN_KEYS = [
        "background_color",
        "background_brightness",
        "blur",
        "mirror",
        "flip",
        "gradient_roll",
    ]

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "bar_color",
                description="Pillar color (ignored when use_gradient is on)",
                default="#22FFAA",
            ): validate_color,
            vol.Optional(
                "count",
                description="Number of pillars along the strip",
                default=6,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=24)),
            vol.Optional(
                "bar_width_pct",
                description="Pillar width as % of available space per pillar (rest is gap)",
                default=60,
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=100)),
            vol.Optional(
                "decay_power",
                description="Attack-release sharpness (higher = punchier, shorter pillar pulse)",
                default=4.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=12.0)),
            vol.Optional(
                "min_brightness",
                description="Brightness floor between beats (0-1)",
                default=0.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "audio_gain",
                description="How much audio power adds to the beat envelope",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(
                "frequency_range",
                description="Frequency range driving the audio gain",
                default="Lows (beat+bass)",
            ): vol.In(list(AudioReactiveEffect.POWER_FUNCS_MAPPING.keys())),
            vol.Optional(
                "chase",
                description="Chase mode — only one pillar lights at a time, advancing each beat",
                default=False,
            ): bool,
            vol.Optional(
                "use_gradient",
                description="Color pillars by index from the gradient",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.brightness = 0.0
        self.power = 0.0
        self.beat_counter = 0
        self.prev_bar_int = 0

    def config_updated(self, config):
        self.bar_color = np.array(
            parse_color(self._config["bar_color"]), dtype=float
        )
        self.count = self._config["count"]
        self.bar_width_pct = self._config["bar_width_pct"]
        self.decay_power = self._config["decay_power"]
        self.min_brightness = self._config["min_brightness"]
        self.audio_gain = self._config["audio_gain"]
        self.power_func = self.POWER_FUNCS_MAPPING[
            self._config["frequency_range"]
        ]
        self.chase = self._config["chase"]
        self.use_gradient = self._config["use_gradient"]

    def audio_data_updated(self, data):
        osc = data.beat_oscillator()
        env = (max(0.0, 1.0 - osc)) ** self.decay_power
        self.brightness = min(
            1.0,
            self.min_brightness
            + env * (1.0 - self.min_brightness),
        )
        self.power = float(getattr(data, self.power_func)())

        if self.chase:
            bar_int = int(data.bar_oscillator())
            if bar_int != self.prev_bar_int:
                self.beat_counter = (self.beat_counter + 1) % max(
                    self.count, 1
                )
            self.prev_bar_int = bar_int

    def render(self):
        self.pixels.fill(0.0)
        if self.count <= 0 or self.pixel_count <= 0:
            return

        slot = self.pixel_count / self.count
        bar_pixels = max(1, int(slot * self.bar_width_pct / 100.0))
        gain = min(1.0, self.brightness + self.power * self.audio_gain)

        for i in range(self.count):
            if self.chase and i != self.beat_counter:
                continue
            if self.use_gradient:
                color = np.array(
                    self.get_gradient_color(i / max(self.count - 1, 1)),
                    dtype=float,
                )
            else:
                color = self.bar_color

            slot_start = int(i * slot)
            slot_mid = slot_start + int(slot / 2)
            start = max(0, slot_mid - bar_pixels // 2)
            end = min(self.pixel_count, start + bar_pixels)
            self.pixels[start:end] = color * gain
