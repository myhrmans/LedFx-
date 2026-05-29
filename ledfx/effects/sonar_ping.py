# fork addition: Sonar Ping — beat-locked band expands outward from the
# center of the strip over the duration of one beat. Fill or Ring modes.
# Geometric, beat-punchy companion to the Pulse effect.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


_MODES = ["Fill", "Ring"]


class SonarPingAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Sonar Ping"
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
                "band_color",
                description="Band color (ignored when use_gradient is on)",
                default="#22CCFF",
            ): validate_color,
            vol.Optional(
                "mode",
                description="Fill (solid band) or Ring (hollow edges)",
                default=_MODES[0],
            ): vol.In(_MODES),
            vol.Optional(
                "initial_width_pct",
                description="Initial band width at beat 0 as % of strip",
                default=4,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=40)),
            vol.Optional(
                "max_expand_pct",
                description="Max expansion as % of half-strip (100 = reaches ends)",
                default=100,
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=100)),
            vol.Optional(
                "ring_thickness_pct",
                description="Ring edge thickness as % of strip (Ring mode only)",
                default=4,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=30)),
            vol.Optional(
                "decay_power",
                description="How fast brightness fades across the beat (higher = sharper ping)",
                default=2.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=10.0)),
            vol.Optional(
                "min_brightness",
                description="Brightness floor between pings (0-1)",
                default=0.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "audio_gain",
                description="How much audio power adds to brightness",
                default=0.3,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
            vol.Optional(
                "frequency_range",
                description="Frequency range driving the audio gain",
                default="Lows (beat+bass)",
            ): vol.In(list(AudioReactiveEffect.POWER_FUNCS_MAPPING.keys())),
            vol.Optional(
                "use_gradient",
                description="Color band by expansion progress from the gradient",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.osc = 0.0
        self.power = 0.0

    def config_updated(self, config):
        self.band_color = np.array(
            parse_color(self._config["band_color"]), dtype=float
        )
        self.mode = self._config["mode"]
        self.initial_width_pct = self._config["initial_width_pct"]
        self.max_expand_pct = self._config["max_expand_pct"]
        self.ring_thickness_pct = self._config["ring_thickness_pct"]
        self.decay_power = self._config["decay_power"]
        self.min_brightness = self._config["min_brightness"]
        self.audio_gain = self._config["audio_gain"]
        self.power_func = self.POWER_FUNCS_MAPPING[
            self._config["frequency_range"]
        ]
        self.use_gradient = self._config["use_gradient"]

    def audio_data_updated(self, data):
        self.osc = data.beat_oscillator()
        self.power = float(getattr(data, self.power_func)())

    def render(self):
        self.pixels.fill(0.0)
        if self.pixel_count <= 0:
            return

        half = self.pixel_count / 2.0
        initial_half = (self.initial_width_pct / 100.0) * self.pixel_count / 2.0
        max_half = (self.max_expand_pct / 100.0) * half
        half_w = initial_half + self.osc * max(max_half - initial_half, 0.0)

        brightness = max(0.0, 1.0 - self.osc) ** self.decay_power
        brightness = min(
            1.0,
            self.min_brightness
            + brightness * (1.0 - self.min_brightness)
            + self.power * self.audio_gain,
        )
        if brightness <= 0:
            return

        if self.use_gradient:
            color = np.array(
                self.get_gradient_color(self.osc), dtype=float
            )
        else:
            color = self.band_color

        center = half
        start = max(0, int(center - half_w))
        end = min(self.pixel_count, int(center + half_w))

        if self.mode == "Fill" or half_w < 2:
            self.pixels[start:end] = color * brightness
            return

        thickness = max(
            1, int(self.pixel_count * self.ring_thickness_pct / 100.0)
        )
        left_end = min(self.pixel_count, start + thickness)
        right_start = max(0, end - thickness)
        self.pixels[start:left_end] = color * brightness
        self.pixels[right_start:end] = color * brightness
