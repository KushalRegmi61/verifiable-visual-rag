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

export type CropStyle = {
  backgroundSize: string;
  backgroundPosition: string;
};

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

/**
 * Background CSS that shows only `bbox`, magnified to fill its container.
 *
 * The window is padded out to `minWidth` by `minHeight` and re-centred on the
 * region, because the thing worth zooming into is usually a single line: 0.0142
 * of the page height against a 0.0312 patch cell. Cropped to its own bounds it
 * magnifies about seventy times and reads as a wall of letterforms with no
 * indication of where on the page it came from. The padding is what makes a
 * crop legible as evidence rather than as a texture.
 *
 * Kept next to toStyle because both consume the same normalized convention and
 * both break the same way if the axes are ever swapped.
 */
/**
 * The padded, page-clamped window a crop actually shows.
 *
 * Separate from toCropStyle because the caller needs it twice: once as the
 * background transform, and once to place the region outline INSIDE the crop.
 * Without that outline a padded crop cannot say which part of what it shows is
 * the evidence and which part is context, which quietly overstates the region.
 */
export function toCropWindow(
  [x0, y0, x1, y1]: BBox,
  {
    minWidth = 0.32,
    minHeight = 0.06,
    aspect,
  }: { minWidth?: number; minHeight?: number; aspect?: number } = {},
): BBox {
  const w = clamp(Math.max(x1 - x0, minWidth), 0, 1);

  // `aspect` is the window's width/height IN NORMALIZED UNITS that fills its
  // container without distortion. Without it, backgroundSize scales the two
  // axes independently: a line region padded to 0.10 of the page height and
  // left at 0.74 of its width was drawn 2.2 times too tall in a 2.35:1
  // container, so the zoom silently restretched the very document it exists to
  // show faithfully.
  //
  // With an aspect, the WIDTH drives the scale and the height follows, because
  // regions are overwhelmingly wide and short and cutting their width is the
  // one thing a crop must never do. If the implied height exceeds the page the
  // window is the whole page height and the container letterboxes, which is
  // honest: no undistorted window that wide can be that tall. An earlier
  // version scaled both axes down to fit instead, and a test here caught it
  // cropping 0.74 of page width down to 0.50 and cutting the region in half.
  const h = aspect && aspect > 0
    ? clamp(w / aspect, 0, 1)
    : clamp(Math.max(y1 - y0, minHeight), 0, 1);

  const left = clamp((x0 + x1) / 2 - w / 2, 0, 1 - w);
  const top = clamp((y0 + y1) / 2 - h / 2, 0, 1 - h);
  return [left, top, left + w, top + h];
}

/**
 * The `aspect` for a crop container, given both aspect ratios as width/height.
 *
 * The page one has to be measured from the loaded image: the retrieval payload
 * carries no page dimensions, and assuming a ratio would reintroduce exactly
 * the distortion this exists to remove.
 */
export function windowAspect(containerAspect: number, pageAspect: number): number {
  return containerAspect / pageAspect;
}

/** `bbox` re-expressed as a fraction of `window`, for drawing inside a crop. */
export function within([x0, y0, x1, y1]: BBox, [wx0, wy0, wx1, wy1]: BBox): BBox {
  const w = wx1 - wx0 || 1;
  const h = wy1 - wy0 || 1;
  return [(x0 - wx0) / w, (y0 - wy0) / h, (x1 - wx0) / w, (y1 - wy0) / h];
}

export function toCropStyle(
  bbox: BBox,
  opts: { minWidth?: number; minHeight?: number; aspect?: number } = {},
): CropStyle {
  const [left, top, right, bottom] = toCropWindow(bbox, opts);
  const w = right - left;
  const h = bottom - top;

  // The denominator is the remainder the background can travel across, not the
  // page. A window filling an axis has nothing to travel, and the expression
  // would be 0/0, so it is pinned rather than left to produce NaN in a style
  // string that the browser then silently drops.
  const px = w >= 1 ? 0 : (left / (1 - w)) * 100;
  const py = h >= 1 ? 0 : (top / (1 - h)) * 100;

  return {
    // `auto` for the height when the caller knows the container ratio: it makes
    // the browser derive the height from the real image, so the crop cannot be
    // distorted by an arithmetic mistake here. Two explicit percentages remain
    // the fallback when the page ratio has not been measured yet.
    backgroundSize: opts.aspect
      ? `${(100 / w).toFixed(4)}% auto`
      : `${(100 / w).toFixed(4)}% ${(100 / h).toFixed(4)}%`,
    backgroundPosition: `${px.toFixed(4)}% ${py.toFixed(4)}%`,
  };
}
