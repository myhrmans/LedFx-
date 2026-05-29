# fork addition: Lightning — on a low-band onset (or strong kick), spawn a
# vertical bolt with a main segment plus 2-3 random offshoot "forks".
# Bolt flashes white at peak and fades to pale blue over ~150ms.
# Designed for tall vertical tubes where the forked geometry reads as a
# real lightning strike.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect


class _Bolt:
    __slots__ = ("segments", "born", "life", "power")

    def __init__(self, segments, born, life, power):
        self.segments = segments
        self.born = born
        self.life = life
        self.power = power


class LightningAudioEffect(AudioReactiveEffect):
    NAME = "Lightning"
    CATEGORY = "Atmospheric"
    HIDDEN_KEYS = ["background_color", "background_brightness", "blur"]

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "bolt_color",
                description="Peak (flash) color",
                default="#FFFFFF",
            ): validate_color,
            vol.Optional(
                "fade_color",
                description="Color the bolt fades into",
                default="#88AAFF",
            ): validate_color,
            vol.Optional(
                "min_bolt_pct",
                description="Min main bolt length as % of strip",
                default=20,
            ): vol.All(vol.Coerce(int), vol.Range(min=5, max=100)),
            vol.Optional(
                "max_bolt_pct",
                description="Max main bolt length as % of strip",
                default=70,
            ): vol.All(vol.Coerce(int), vol.Range(min=10, max=100)),
            vol.Optional(
                "life_ms",
                description="How long a bolt lives in ms",
                default=180,
            ): vol.All(vol.Coerce(int), vol.Range(min=40, max=800)),
            vol.Optional(
                "kick_threshold",
                description="Low-band level required to strike (also requires onset)",
                default=0.45,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=2.0)),
            vol.Optional(
                "min_strike_interval_ms",
                description="Minimum gap between strikes in ms",
                default=80,
            ): vol.All(vol.Coerce(int), vol.Range(min=20, max=2000)),
            vol.Optional(
                "max_simultaneous",
                description="Maximum simultaneous bolts",
                default=4,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
            vol.Optional(
                "fork_count",
                description="Approx number of side forks per bolt (random 1..n)",
                default=3,
            ): vol.All(vol.Coerce(int), vol.Range(min=0, max=8)),
        }
    )

    def on_activate(self, pixel_count):
        self.bolts = []
        self.prev_lows = 0.0
        self.last_strike = 0.0
        self._lows_filter = self.create_filter(
            alpha_decay=0.1, alpha_rise=0.99
        )
        self._rng = np.random.default_rng()

    def config_updated(self, config):
        self.bolt_color = np.array(
            parse_color(self._config["bolt_color"]), dtype=float
        )
        self.fade_color = np.array(
            parse_color(self._config["fade_color"]), dtype=float
        )
        self.min_bolt_pct = self._config["min_bolt_pct"]
        self.max_bolt_pct = max(
            self._config["max_bolt_pct"], self._config["min_bolt_pct"]
        )
        self.life_s = self._config["life_ms"] / 1000.0
        self.kick_threshold = self._config["kick_threshold"]
        self.min_strike_interval_s = (
            self._config["min_strike_interval_ms"] / 1000.0
        )
        self.max_simultaneous = self._config["max_simultaneous"]
        self.fork_count = self._config["fork_count"]

    def _spawn_bolt(self, power):
        min_px = max(1, int(self.pixel_count * self.min_bolt_pct / 100))
        max_px = max(min_px + 1, int(self.pixel_count * self.max_bolt_pct / 100))
        # Power scales toward the longer end of the range.
        scale = min(power / 1.5, 1.0)
        main_len = int(min_px + scale * (max_px - min_px))
        main_len = max(1, min(main_len, self.pixel_count))

        root = int(self._rng.integers(0, max(self.pixel_count - main_len, 1)))
        segments = [(root, root + main_len)]

        n_forks = (
            int(self._rng.integers(1, self.fork_count + 1))
            if self.fork_count > 0
            else 0
        )
        for _ in range(n_forks):
            anchor = int(self._rng.integers(root, root + main_len))
            fork_len = max(2, int(self._rng.integers(2, max(3, main_len // 3 + 1))))
            direction = 1 if self._rng.random() < 0.5 else -1
            a = anchor + direction * fork_len
            start, end = min(anchor, a), max(anchor, a)
            start = max(0, start)
            end = min(self.pixel_count, end + 1)
            if end > start:
                segments.append((start, end))

        self.bolts.append(
            _Bolt(
                segments=segments,
                born=self.now,
                life=self.life_s,
                power=power,
            )
        )

    def audio_data_updated(self, data):
        lows = self._lows_filter.update(
            float(np.mean(data.lows_power(filtered=False)))
        )
        # Strike on a rising edge across threshold, gated by min interval and
        # the aubio onset detector to avoid mid-sustain triggers.
        if (
            data.onset()
            and lows > self.kick_threshold
            and self.prev_lows <= self.kick_threshold
            and self.now - self.last_strike >= self.min_strike_interval_s
            and len(self.bolts) < self.max_simultaneous
        ):
            self._spawn_bolt(min(lows, 2.0))
            self.last_strike = self.now
        self.prev_lows = lows

    def render(self):
        self.pixels.fill(0.0)

        alive = []
        for bolt in self.bolts:
            age = self.now - bolt.born
            if age >= bolt.life:
                continue
            # Sharp flash at birth, exponential-ish fade to fade_color.
            t = age / bolt.life
            flash = (1.0 - t) ** 2
            color = self.bolt_color * flash + self.fade_color * (1.0 - flash) * (1.0 - t)
            for start, end in bolt.segments:
                if end > start:
                    self.pixels[start:end] += color
            alive.append(bolt)
        self.bolts = alive

        np.minimum(self.pixels, 255.0, out=self.pixels)
