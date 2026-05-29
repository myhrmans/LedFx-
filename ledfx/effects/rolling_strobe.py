# fork addition: Rolling Strobe — ladder of N short strobe rungs along the
# strip. Each rung flashes on the beat with a phase offset of i/N, so within
# one beat the flashes roll across the strip like a wagon wheel. The
# rolls_per_unit config lets you stretch the rolling cycle across multiple
# beats (or compress it into a sub-beat) for different drop-vs-build feels.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


class RollingStrobeAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Rolling Strobe"
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
                "rung_color",
                description="Strobe color (ignored when use_gradient is on)",
                default="#FFFFFF",
            ): validate_color,
            vol.Optional(
                "count",
                description="Number of strobe rungs along the strip",
                default=8,
            ): vol.All(vol.Coerce(int), vol.Range(min=2, max=32)),
            vol.Optional(
                "rung_width_pct",
                description="Rung width as % of the gap between rungs",
                default=35,
            ): vol.All(vol.Coerce(int), vol.Range(min=5, max=100)),
            vol.Optional(
                "flash_tail",
                description="How long the flash lingers within its slice (0-1)",
                default=0.25,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.05, max=1.0)),
            vol.Optional(
                "decay_power",
                description="Flash decay shape (higher = harder flash, shorter linger)",
                default=2.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=1.0, max=10.0)),
            vol.Optional(
                "rolls_per_unit",
                description="Number of full rolls per unit (1 = roll every beat, 4 = roll every bar)",
                default=1,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.25, max=8.0)),
            vol.Optional(
                "roll_unit",
                description="Time unit the roll is measured in",
                default="Beat",
            ): vol.In(["Beat", "Bar"]),
            vol.Optional(
                "direction",
                description="Roll direction along the strip",
                default="Forward",
            ): vol.In(["Forward", "Reverse"]),
            vol.Optional(
                "audio_gain",
                description="How much audio power gates the strobe (0 = always on, higher = needs energy)",
                default=0.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=1.0)),
            vol.Optional(
                "frequency_range",
                description="Frequency range driving the audio gate",
                default="Lows (beat+bass)",
            ): vol.In(list(AudioReactiveEffect.POWER_FUNCS_MAPPING.keys())),
            vol.Optional(
                "use_gradient",
                description="Color rungs by index from the gradient",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.phase = 0.0
        self.gate = 1.0

    def config_updated(self, config):
        self.rung_color = np.array(
            parse_color(self._config["rung_color"]), dtype=float
        )
        self.count = self._config["count"]
        self.rung_width_pct = self._config["rung_width_pct"]
        self.flash_tail = self._config["flash_tail"]
        self.decay_power = self._config["decay_power"]
        self.rolls_per_unit = self._config["rolls_per_unit"]
        self.roll_unit = self._config["roll_unit"]
        self.direction = 1 if self._config["direction"] == "Forward" else -1
        self.audio_gain = self._config["audio_gain"]
        self.power_func = self.POWER_FUNCS_MAPPING[
            self._config["frequency_range"]
        ]
        self.use_gradient = self._config["use_gradient"]

    def audio_data_updated(self, data):
        if self.roll_unit == "Bar":
            unit_pos = data.bar_oscillator() / 4.0
        else:
            unit_pos = data.beat_oscillator()
        self.phase = (unit_pos * self.rolls_per_unit) % 1.0

        if self.audio_gain > 0:
            power = float(getattr(data, self.power_func)())
            self.gate = min(1.0, max(0.0, 1.0 - self.audio_gain + power))
        else:
            self.gate = 1.0

    def render(self):
        self.pixels.fill(0.0)
        if self.count <= 0 or self.pixel_count <= 0 or self.gate <= 0:
            return

        slot = self.pixel_count / self.count
        rung_pixels = max(1, int(slot * self.rung_width_pct / 100.0))

        for i in range(self.count):
            # Phase offset by index — rungs flash one after another within
            # one rolling cycle. Direction flips offset sign.
            offset = (i / self.count) * self.direction
            phi = (self.phase - offset) % 1.0
            if phi > self.flash_tail:
                brightness = 0.0
            else:
                brightness = (1.0 - phi / self.flash_tail) ** self.decay_power
            brightness *= self.gate
            if brightness <= 0:
                continue

            if self.use_gradient:
                color = np.array(
                    self.get_gradient_color(i / max(self.count - 1, 1)),
                    dtype=float,
                )
            else:
                color = self.rung_color

            slot_start = int(i * slot)
            slot_mid = slot_start + int(slot / 2)
            start = max(0, slot_mid - rung_pixels // 2)
            end = min(self.pixel_count, start + rung_pixels)
            self.pixels[start:end] = color * brightness
