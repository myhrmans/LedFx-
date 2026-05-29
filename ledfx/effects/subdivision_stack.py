# fork addition: Subdivision Stack — strip divided into N stacked zones,
# each zone flashes at its own subdivision rate (whole beat, 1/8, 1/16, ...)
# with attack-release envelope. Bottom zone runs the slowest (whole beats)
# and each zone above doubles the rate, visualizing the rhythmic grid.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


class SubdivisionStackAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Subdivision Stack"
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
                "zone_color",
                description="Zone color (ignored when use_gradient is on)",
                default="#FFAA22",
            ): validate_color,
            vol.Optional(
                "stacks",
                description="Number of subdivision zones along the strip",
                default=4,
            ): vol.All(vol.Coerce(int), vol.Range(min=2, max=6)),
            vol.Optional(
                "bottom_subdivision",
                description="Flashes-per-beat of the bottom zone (each zone above doubles)",
                default=1,
            ): vol.In([1, 2, 4]),
            vol.Optional(
                "decay_power",
                description="Attack-release sharpness (higher = punchier flashes)",
                default=3.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=12.0)),
            vol.Optional(
                "min_brightness",
                description="Brightness floor between flashes (0-1)",
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
                "gap_pct",
                description="Gap between zones as % of zone height (0 = touching)",
                default=8,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=50)),
            vol.Optional(
                "use_gradient",
                description="Color zones from the gradient (bottom = start, top = end)",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.osc = 0.0
        self.power = 0.0

    def config_updated(self, config):
        self.zone_color = np.array(
            parse_color(self._config["zone_color"]), dtype=float
        )
        self.stacks = self._config["stacks"]
        self.bottom_subdivision = self._config["bottom_subdivision"]
        self.decay_power = self._config["decay_power"]
        self.min_brightness = self._config["min_brightness"]
        self.audio_gain = self._config["audio_gain"]
        self.power_func = self.POWER_FUNCS_MAPPING[
            self._config["frequency_range"]
        ]
        self.gap_pct = self._config["gap_pct"]
        self.use_gradient = self._config["use_gradient"]
        self.subdivisions = [
            self.bottom_subdivision * (2 ** i) for i in range(self.stacks)
        ]

    def audio_data_updated(self, data):
        self.osc = data.beat_oscillator()
        self.power = float(getattr(data, self.power_func)())

    def render(self):
        self.pixels.fill(0.0)
        if self.stacks <= 0 or self.pixel_count <= 0:
            return

        slot = self.pixel_count / self.stacks
        gap_px = int(slot * self.gap_pct / 100.0 / 2)
        audio_boost = self.power * self.audio_gain

        for i in range(self.stacks):
            sub = self.subdivisions[i]
            phi = (self.osc * sub) % 1.0
            env = (max(0.0, 1.0 - phi)) ** self.decay_power
            brightness = min(
                1.0,
                self.min_brightness
                + env * (1.0 - self.min_brightness)
                + audio_boost,
            )
            if brightness <= 0:
                continue

            if self.use_gradient:
                color = np.array(
                    self.get_gradient_color(i / max(self.stacks - 1, 1)),
                    dtype=float,
                )
            else:
                color = self.zone_color

            slot_start = int(i * slot) + gap_px
            slot_end = int((i + 1) * slot) - gap_px
            slot_start = max(0, slot_start)
            slot_end = min(self.pixel_count, slot_end)
            if slot_end > slot_start:
                self.pixels[slot_start:slot_end] = color * brightness
