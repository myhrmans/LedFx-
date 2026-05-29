# fork addition: EKG — hospital-monitor style heartbeat line. A "pen" travels
# along the strip at a beat-locked speed (one full traverse per bar). At each
# beat it writes a sharp "lub" spike, then a smaller "dub" mid-beat. Old
# values fade exponentially so the recent history is visible behind the pen.
import numpy as np
import voluptuous as vol

from ledfx.color import parse_color, validate_color
from ledfx.effects.audio import AudioReactiveEffect


class EkgAudioEffect(AudioReactiveEffect):
    NAME = "EKG"
    CATEGORY = "BPM"
    HIDDEN_KEYS = [
        "background_color",
        "background_brightness",
        "blur",
        "mirror",
        "flip",
    ]

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "trace_color",
                description="Color of the EKG trace",
                default="#00FF66",
            ): validate_color,
            vol.Optional(
                "head_color",
                description="Color of the bright pen marker",
                default="#FFFFFF",
            ): validate_color,
            vol.Optional(
                "bars_per_traverse",
                description="How many bars (4 beats each) for the pen to cross the strip once",
                default=1,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=8)),
            vol.Optional(
                "lub_width",
                description="Width of the strong 'lub' spike (smaller = sharper)",
                default=0.06,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=0.3)),
            vol.Optional(
                "dub_width",
                description="Width of the secondary 'dub' spike",
                default=0.08,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.01, max=0.3)),
            vol.Optional(
                "dub_offset",
                description="Where in the beat the 'dub' lands (0-1, fraction of beat)",
                default=0.35,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=0.9)),
            vol.Optional(
                "dub_height",
                description="Relative amplitude of dub vs lub",
                default=0.55,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.1, max=1.0)),
            vol.Optional(
                "baseline",
                description="Baseline brightness when no spike (0-1)",
                default=0.06,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=0.3)),
            vol.Optional(
                "fade_rate",
                description="How fast old trace pixels fade per second (higher = shorter tail)",
                default=2.5,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.3, max=15.0)),
            vol.Optional(
                "head_size",
                description="Pen head size in pixels",
                default=2,
            ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
            vol.Optional(
                "audio_gain",
                description="How much bass adds to the lub amplitude",
                default=0.3,
            ): vol.All(vol.Coerce(float), vol.Range(min=0.0, max=2.0)),
        }
    )

    def on_activate(self, pixel_count):
        self.last_pen_pos = 0
        self.power = 0.0
        self.bar_osc = 0.0

    def config_updated(self, config):
        self.trace_color = np.array(
            parse_color(self._config["trace_color"]), dtype=float
        )
        self.head_color = np.array(
            parse_color(self._config["head_color"]), dtype=float
        )
        self.bars_per_traverse = self._config["bars_per_traverse"]
        self.lub_width = self._config["lub_width"]
        self.dub_width = self._config["dub_width"]
        self.dub_offset = self._config["dub_offset"]
        self.dub_height = self._config["dub_height"]
        self.baseline = self._config["baseline"]
        self.fade_rate = self._config["fade_rate"]
        self.head_size = self._config["head_size"]
        self.audio_gain = self._config["audio_gain"]

    def audio_data_updated(self, data):
        self.bar_osc = data.bar_oscillator()
        self.power = float(data.bass_power())

    def _ekg_value(self, beat_phase):
        # beat_phase in [0, 1). Sum of two Gaussian-ish bumps.
        def bump(phase, center, width):
            x = (phase - center) / width
            return float(np.exp(-x * x))

        lub = bump(beat_phase, 0.0, self.lub_width)
        # wrap-aware lub: if phase is near 1, treat as near 0 as well
        lub = max(lub, bump(beat_phase, 1.0, self.lub_width))
        dub = self.dub_height * bump(
            beat_phase, self.dub_offset, self.dub_width
        )
        return self.baseline + (lub + dub) * (1.0 + self.power * self.audio_gain)

    def render(self):
        dt = max(self.passed, 0.0)
        decay = float(np.exp(-self.fade_rate * dt))
        self.pixels *= decay

        # pen position: bar_oscillator advances 0->4 per bar; traversal
        # covers self.bars_per_traverse bars total.
        traverse_units = self.bars_per_traverse * 4.0
        traverse_phase = (self.bar_osc / traverse_units) % 1.0
        pen_pos = int(traverse_phase * self.pixel_count)

        # Write values along the path from last_pen_pos to pen_pos, picking up
        # the corresponding beat_phase for each in-between pixel so a faster
        # pen still renders the shape correctly. If we wrapped around, split
        # the write into two segments.
        if pen_pos >= self.last_pen_pos:
            self._draw_segment(self.last_pen_pos, pen_pos, traverse_units)
        else:
            self._draw_segment(
                self.last_pen_pos, self.pixel_count, traverse_units
            )
            self._draw_segment(0, pen_pos, traverse_units)

        # Pen head — bright marker at current position.
        end = min(pen_pos + self.head_size, self.pixel_count)
        self.pixels[pen_pos:end] = self.head_color

        self.last_pen_pos = pen_pos
        np.minimum(self.pixels, 255.0, out=self.pixels)

    def _draw_segment(self, start, end, traverse_units):
        if end <= start:
            return
        for px in range(start, end):
            traverse_phase = (px / max(self.pixel_count, 1))
            bar_phase = traverse_phase * traverse_units
            beat_phase = bar_phase % 1.0
            value = self._ekg_value(beat_phase)
            self.pixels[px] = self.trace_color * min(value, 1.0)
