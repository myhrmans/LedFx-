# fork addition: Pulse — entire strip brightness oscillates with the beat.
# The simplest possible beat-sync effect — uniform color or gradient,
# selectable sine ("breathing") or attack-release ("punch") envelope.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


_SHAPES = ["Sine (breathing)", "Attack-release (punch)"]


class PulseAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Pulse"
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
                "pulse_color",
                description="Pulse color (ignored when use_gradient is on)",
                default="#FF2266",
            ): validate_color,
            vol.Optional(
                "shape",
                description="Envelope shape across the beat",
                default=_SHAPES[0],
            ): vol.In(_SHAPES),
            vol.Optional(
                "min_brightness",
                description="Brightness floor between pulses (0-1)",
                default=0.1,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "decay_power",
                description="Attack-release sharpness (higher = punchier)",
                default=3.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=10.0)),
            vol.Optional(
                "audio_gain",
                description="How much audio power adds on top of the beat envelope",
                default=0.4,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(
                "frequency_range",
                description="Frequency range driving the audio gain",
                default="Lows (beat+bass)",
            ): vol.In(list(AudioReactiveEffect.POWER_FUNCS_MAPPING.keys())),
            vol.Optional(
                "use_gradient",
                description="Cycle color through the gradient across each bar",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.brightness = 0.0
        self.color = self.pulse_color

    def config_updated(self, config):
        self.pulse_color = np.array(
            parse_color(self._config["pulse_color"]), dtype=float
        )
        self.shape = self._config["shape"]
        self.min_brightness = self._config["min_brightness"]
        self.decay_power = self._config["decay_power"]
        self.audio_gain = self._config["audio_gain"]
        self.power_func = self.POWER_FUNCS_MAPPING[
            self._config["frequency_range"]
        ]
        self.use_gradient = self._config["use_gradient"]
        self.color = self.pulse_color

    def audio_data_updated(self, data):
        osc = data.beat_oscillator()
        if self.shape == _SHAPES[0]:
            env = 0.5 + 0.5 * np.cos(2.0 * np.pi * osc)
        else:
            env = (max(0.0, 1.0 - osc)) ** self.decay_power

        power = float(getattr(data, self.power_func)())
        self.brightness = min(
            1.0,
            self.min_brightness
            + env * (1.0 - self.min_brightness)
            + power * self.audio_gain,
        )

        if self.use_gradient:
            bar = data.bar_oscillator()
            self.color = np.array(
                self.get_gradient_color((bar / 4.0) % 1.0), dtype=float
            )

    def render(self):
        self.pixels[:] = self.color * self.brightness
