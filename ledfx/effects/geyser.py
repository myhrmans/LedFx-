# fork addition: Geyser — bass kicks launch particles from the bottom of the
# strip. Each particle rises with kick-proportional velocity, decelerates
# under gravity, fades over its lifetime. A bottom "glow zone" tracks
# sustained low-band energy.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect
from ledfx.effects.gradient import GradientEffect


class _Particle:
    __slots__ = ("pos", "vel", "life", "born", "color")

    def __init__(self, pos, vel, life, born, color):
        self.pos = pos
        self.vel = vel
        self.life = life
        self.born = born
        self.color = color


class GeyserAudioEffect(AudioReactiveEffect, GradientEffect):
    NAME = "Geyser"
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
                "particle_color",
                description="Color of launched particles",
                default="#FF4400",
            ): validate_color,
            vol.Optional(
                "glow_color",
                description="Color of the bottom glow zone",
                default="#FF2200",
            ): validate_color,
            vol.Optional(
                "kick_threshold",
                description="Low-band level required to launch a particle",
                default=0.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=2.0)),
            vol.Optional(
                "velocity_gain",
                description="Initial particle velocity gain (% strip/sec per unit kick)",
                default=140,
            ): vol.All(vol.Coerce(int), vol.Range(min=20, max=400)),
            vol.Optional(
                "gravity",
                description="Gravity in % strip / sec^2 (pulls particles back down)",
                default=180,
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=600)),
            vol.Optional(
                "particle_life",
                description="Particle lifetime in seconds",
                default=1.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.2, max=5.0)),
            vol.Optional(
                "particle_size",
                description="Particle render size in pixels",
                default=3,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
            vol.Optional(
                "max_particles",
                description="Max simultaneous particles",
                default=12,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=40)),
            vol.Optional(
                "glow_height",
                description="Bottom glow height as % of strip",
                default=15,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=60)),
            vol.Optional(
                "use_gradient",
                description="Color particles by launch power from the gradient",
                default=False,
            ): bool,
        }
    )

    def on_activate(self, pixel_count):
        self.particles = []
        self.bottom_glow = 0.0
        self.prev_lows = 0.0
        self._lows_filter = self.create_filter(
            alpha_decay=0.08, alpha_rise=0.95
        )

    def config_updated(self, config):
        self.particle_color = np.array(
            parse_color(self._config["particle_color"]), dtype=float
        )
        self.glow_color = np.array(
            parse_color(self._config["glow_color"]), dtype=float
        )
        self.kick_threshold = self._config["kick_threshold"]
        self.velocity_gain = self._config["velocity_gain"]
        self.gravity = self._config["gravity"]
        self.particle_life = self._config["particle_life"]
        self.particle_size = self._config["particle_size"]
        self.max_particles = self._config["max_particles"]
        self.glow_pct = self._config["glow_height"]
        self.use_gradient = self._config["use_gradient"]

    def audio_data_updated(self, data):
        lows = self._lows_filter.update(
            float(np.mean(data.lows_power(filtered=False)))
        )
        # Rising edge across threshold = "kick" — only fires once per kick.
        if (
            lows > self.kick_threshold
            and self.prev_lows <= self.kick_threshold
            and len(self.particles) < self.max_particles
        ):
            power = min(lows, 2.0)
            if self.use_gradient:
                color = np.array(
                    self.get_gradient_color(min(power / 2.0, 1.0)),
                    dtype=float,
                )
            else:
                color = self.particle_color
            self.particles.append(
                _Particle(
                    pos=0.0,
                    vel=power * (self.velocity_gain / 100.0)
                    * max(self.pixel_count, 1),
                    life=self.particle_life,
                    born=self.now,
                    color=color,
                )
            )
        self.prev_lows = lows
        self.bottom_glow = lows

    def render(self):
        self.pixels.fill(0.0)

        dt = max(self.passed, 0.0)
        gravity_px = (self.gravity / 100.0) * max(self.pixel_count, 1)

        # Integrate + render particles; drop dead ones in place.
        alive = []
        for p in self.particles:
            p.vel -= gravity_px * dt
            p.pos += p.vel * dt
            age = self.now - p.born
            if age >= p.life or p.pos < 0:
                continue
            brightness = max(0.0, 1.0 - age / p.life)
            pos_int = int(p.pos)
            end = min(pos_int + self.particle_size, self.pixel_count)
            if pos_int < self.pixel_count and end > 0:
                start = max(pos_int, 0)
                self.pixels[start:end] += p.color * brightness
            alive.append(p)
        self.particles = alive

        # Bottom glow tracks sustained low-band energy, with a falloff
        # gradient toward the top of the glow zone.
        if self.glow_pct > 0 and self.bottom_glow > 0:
            glow_h = max(1, int(self.pixel_count * self.glow_pct / 100))
            falloff = np.linspace(1.0, 0.0, glow_h, dtype=float)[:, None]
            self.pixels[:glow_h] += (
                self.glow_color * self.bottom_glow * falloff
            )

        np.minimum(self.pixels, 255.0, out=self.pixels)
