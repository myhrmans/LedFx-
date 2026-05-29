# fork addition: Comet — single bright head with an exponentially fading
# tail. Built for tall vertical tubes where a clean traveling head + trail
# reads well. Head speed scales with audio power; optionally bounces or
# wraps and respawns on each bar.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


class CometAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Comet"
    CATEGORY = "Classic"
    HIDDEN_KEYS = [
        "background_color",
        "background_brightness",
        "blur",
        "gradient_roll",
    ]

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "head_color",
                description="Color of the comet head",
                default="#FFFFFF",
            ): validate_color,
            vol.Optional(
                "base_speed",
                description="Base head speed as % of strip per second",
                default=80,
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=400)),
            vol.Optional(
                "audio_speed_gain",
                description="How much audio power adds to head speed (0 = constant)",
                default=1.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=5.0)),
            vol.Optional(
                "tail_decay",
                description="Tail decay rate per second (higher = shorter tail)",
                default=4.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=20.0)),
            vol.Optional(
                "head_size",
                description="Head size in pixels",
                default=3,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
            vol.Optional(
                "frequency_range",
                description="Frequency range driving head speed",
                default="Lows (beat+bass)",
            ): vol.In(list(AudioReactiveEffect.POWER_FUNCS_MAPPING.keys())),
            vol.Optional(
                "bounce",
                description="Bounce off the ends instead of wrapping",
                default=False,
            ): bool,
            vol.Optional(
                "respawn_on_bar",
                description="Reset head to start of strip on each new bar",
                default=False,
            ): bool,
            vol.Optional(
                "use_gradient_head",
                description="Color the head from the gradient instead of head_color",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.head_pos = 0.0
        self.direction = 1
        self.power = 0.0
        self.last_bar = -1

    def config_updated(self, config):
        self.head_color = np.array(
            parse_color(self._config["head_color"]), dtype=float
        )
        self.power_func = self.POWER_FUNCS_MAPPING[
            self._config["frequency_range"]
        ]
        self.base_speed = self._config["base_speed"]
        self.audio_speed_gain = self._config["audio_speed_gain"]
        self.tail_decay = self._config["tail_decay"]
        self.head_size = self._config["head_size"]
        self.bounce = self._config["bounce"]
        self.respawn_on_bar = self._config["respawn_on_bar"]
        self.use_gradient_head = self._config["use_gradient_head"]

    def audio_data_updated(self, data):
        self.power = getattr(data, self.power_func)()
        if self.respawn_on_bar:
            bar = int(data.bar_oscillator())
            if bar != self.last_bar:
                self.last_bar = bar
                self.head_pos = (
                    0.0
                    if self.direction == 1
                    else float(max(self.pixel_count - 1, 0))
                )

    def render(self):
        # Frame-rate-independent exponential decay of the existing pixel
        # buffer — yesterday's head becomes today's tail. Relies on the
        # base Effect class preserving self.pixels between frames.
        decay = float(np.exp(-self.tail_decay * max(self.passed, 0.0)))
        self.pixels *= decay

        speed_pct = self.base_speed * (
            1.0 + self.audio_speed_gain * self.power
        )
        step = (self.pixel_count / 100.0) * speed_pct * self.passed
        self.head_pos += step * self.direction

        if self.bounce:
            upper = max(self.pixel_count - self.head_size, 0)
            if self.head_pos >= upper:
                self.head_pos = float(upper)
                self.direction = -1
            elif self.head_pos <= 0:
                self.head_pos = 0.0
                self.direction = 1
        else:
            self.head_pos %= self.pixel_count

        if self.use_gradient_head:
            head_color = self.get_gradient_color(
                (self.head_pos / max(self.pixel_count, 1)) % 1
            )
        else:
            head_color = self.head_color

        head_int = int(self.head_pos)
        head_end = min(head_int + self.head_size, self.pixel_count)
        self.pixels[head_int:head_end] = head_color

        if not self.bounce:
            wrap = head_int + self.head_size - self.pixel_count
            if wrap > 0:
                self.pixels[:wrap] = head_color
