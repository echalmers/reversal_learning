from matplotlib import colors
import colorsys


def scale_luminosity(named_color: str, scale: float) -> tuple:
    hls = colorsys.rgb_to_hls(*colors.to_rgb(named_color))
    rgb = list(colorsys.hls_to_rgb(hls[0], hls[1] * scale, hls[2]))
    return tuple([round(x, 3) for x in rgb])
