# fork addition: Rainfall — droplets spawn on hi-hat onsets at the top of
# the strip, fall under real gravity (accelerating), leave persistence trails,
# splash on impact at the bottom, and feed a slowly-decaying bottom "pool".
# Distinct from the existing Rain effect which stamps sprites without physics.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


class _Drop:
    __slots__ = ("pos", "vel", "color")

    def __init__(self, pos, vel, color):
        self.pos = pos
        self.vel = vel
        self.color = color


class RainfallAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Rainfall"
    CATEGORY = "Atmospheric"
    HIDDEN_KEYS = [
        "background_color",
        "background_brightness",
        "blur",
        "gradient_roll",
    ]

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "drop_color",
                description="Color of droplets and trails",
                default="#88CCFF",
            ): validate_color,
            vol.Optional(
                "splash_color",
                description="Color of the splash flash at the bottom",
                default="#FFFFFF",
            ): validate_color,
            vol.Optional(
                "pool_color",
                description="Color of the accumulating bottom pool",
                default="#446699",
            ): validate_color,
            vol.Optional(
                "spawn_threshold",
                description="High-band level required to spawn a droplet",
                default=0.35,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=2.0)),
            vol.Optional(
                "initial_velocity",
                description="Initial fall velocity as % strip/sec",
                default=40,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=300)),
            vol.Optional(
                "gravity",
                description="Gravity as % strip / sec^2",
                default=250,
            ): vol.All(vol.Coerce(int), vol.Range(min=20, max=800)),
            vol.Optional(
                "trail_decay",
                description="Trail decay rate per second (higher = shorter trail)",
                default=6.0,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.5, max=20.0)),
            vol.Optional(
                "max_drops",
                description="Max simultaneous droplets",
                default=20,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=60)),
            vol.Optional(
                "splash_size",
                description="Splash flash size in pixels",
                default=5,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=20)),
            vol.Optional(
                "splash_life_ms",
                description="Splash flash duration in ms",
                default=140,
            ): vol.All(vol.Coerce(int), vol.Range(min=20, max=600)),
            vol.Optional(
                "pool_max_height",
                description="Maximum bottom pool height in % of strip",
                default=18,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=60)),
            vol.Optional(
                "pool_decay",
                description="How fast the bottom pool drains per second (units of pool height)",
                default=0.6,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.05, max=5.0)),
            vol.Optional(
                "use_gradient",
                description="Color droplets from gradient (cycles per spawn)",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.drops = []
        self.splashes = []
        self.pool = 0.0
        self.prev_highs = 0.0
        self._highs_filter = self.create_filter(
            alpha_decay=0.12, alpha_rise=0.99
        )
        self._spawn_count = 0

    def config_updated(self, config):
        self.drop_color = np.array(
            parse_color(self._config["drop_color"]), dtype=float
        )
        self.splash_color = np.array(
            parse_color(self._config["splash_color"]), dtype=float
        )
        self.pool_color = np.array(
            parse_color(self._config["pool_color"]), dtype=float
        )
        self.spawn_threshold = self._config["spawn_threshold"]
        self.initial_velocity = self._config["initial_velocity"]
        self.gravity = self._config["gravity"]
        self.trail_decay = self._config["trail_decay"]
        self.max_drops = self._config["max_drops"]
        self.splash_size = self._config["splash_size"]
        self.splash_life_s = self._config["splash_life_ms"] / 1000.0
        self.pool_max_pct = self._config["pool_max_height"]
        self.pool_decay = self._config["pool_decay"]
        self.use_gradient = self._config["use_gradient"]

    def audio_data_updated(self, data):
        highs = self._highs_filter.update(
            float(np.mean(data.high_power(filtered=False)))
        )
        if (
            highs > self.spawn_threshold
            and self.prev_highs <= self.spawn_threshold
            and len(self.drops) < self.max_drops
        ):
            if self.use_gradient:
                self._spawn_count += 1
                color = np.array(
                    self.get_gradient_color((self._spawn_count * 0.137) % 1.0),
                    dtype=float,
                )
            else:
                color = self.drop_color
            top = float(max(self.pixel_count - 1, 0))
            self.drops.append(
                _Drop(
                    pos=top,
                    vel=-(self.initial_velocity / 100.0) * max(self.pixel_count, 1),
                    color=color,
                )
            )
        self.prev_highs = highs

    def render(self):
        dt = max(self.passed, 0.0)
        # Persistence-based trails: previous frame fades exponentially.
        decay = float(np.exp(-self.trail_decay * dt))
        self.pixels *= decay

        gravity_px = (self.gravity / 100.0) * max(self.pixel_count, 1)

        alive = []
        for d in self.drops:
            d.vel -= gravity_px * dt
            d.pos += d.vel * dt
            if d.pos <= 0:
                self.splashes.append((self.now, d.color))
                self.pool = min(
                    self.pool + 0.18, self.pool_max_pct / 100.0
                )
                continue
            pos_int = int(d.pos)
            if 0 <= pos_int < self.pixel_count:
                self.pixels[pos_int] = np.maximum(
                    self.pixels[pos_int], d.color
                )
            alive.append(d)
        self.drops = alive

        # Splash flashes — short-lived bright stamps at the bottom.
        live_splashes = []
        for born, scolor in self.splashes:
            age = self.now - born
            if age >= self.splash_life_s:
                continue
            t = age / self.splash_life_s
            brightness = (1.0 - t) ** 2
            blend = self.splash_color * brightness + scolor * (1.0 - brightness) * brightness
            end = min(self.splash_size, self.pixel_count)
            self.pixels[:end] = np.maximum(self.pixels[:end], blend)
            live_splashes.append((born, scolor))
        self.splashes = live_splashes

        # Bottom pool — falloff gradient toward the top of the pool.
        self.pool = max(0.0, self.pool - self.pool_decay * dt * 0.1)
        if self.pool > 0:
            pool_h = max(1, int(self.pool * self.pixel_count))
            pool_h = min(pool_h, self.pixel_count)
            falloff = np.linspace(1.0, 0.0, pool_h, dtype=float)[:, None]
            self.pixels[:pool_h] = np.maximum(
                self.pixels[:pool_h], self.pool_color * falloff
            )

        np.minimum(self.pixels, 255.0, out=self.pixels)
