// Regions arrive as normalized 0-1 boxes against the displayed page rect,
// origin top-left. Percentages rescale for free at any display size, which is
// why S1 normalized the convention in the first place, and why this draws divs
// rather than a canvas.

export type BBox = [number, number, number, number];

export type Region = {
  bbox: BBox;
  score: number;
  modality: "visual" | "text";
  resolution: "line" | "block" | null;
  text: string | null;
};

export type BoxStyle = {
  left: string;
  top: string;
  width: string;
  height: string;
};

// No rounding. A line box on a real page measures 0.0142 of the page height,
// so rounding to whole percentages would give it zero height and draw nothing.
export function toStyle([x0, y0, x1, y1]: BBox): BoxStyle {
  return {
    left: `${x0 * 100}%`,
    top: `${y0 * 100}%`,
    width: `${(x1 - x0) * 100}%`,
    height: `${(y1 - y0) * 100}%`,
  };
}
