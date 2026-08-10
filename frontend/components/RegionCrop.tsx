import type { CSSProperties } from "react";
import {
  toCropStyle,
  toCropWindow,
  toStyle,
  windowAspect,
  within,
  type BBox,
} from "@/lib/overlay";

type Props = {
  imageUrl: string;
  bbox: BBox;
  colour: string;
  className?: string;
  minWidth?: number;
  minHeight?: number;
  /** Width/height of the container in CSS pixels. */
  containerAspect: number;
  /** Merged into the container, which is where the aspect box is set. */
  style?: CSSProperties;
  /**
   * Width/height of the page image, measured from the loaded element. Null
   * until the first load, and the crop then falls back to a stretched window
   * rather than guessing a ratio.
   */
  pageAspect: number | null;
};

/**
 * A magnified window onto one region of the page.
 *
 * The exact region is outlined inside the window. A crop is padded out to stay
 * legible, so without the outline the user would read the whole padded window
 * as the evidence, which claims more ground than the system actually grounded.
 * The padding is context; the outline is the claim.
 *
 * Drawn as a background rather than a cropped <img> so the browser reuses the
 * page image it has already fetched and decoded for the main viewer, rather
 * than holding a second copy per claim.
 */
export function RegionCrop({
  imageUrl,
  bbox,
  colour,
  className = "",
  minWidth,
  minHeight,
  containerAspect,
  pageAspect,
  style,
}: Props) {
  const opts = {
    minWidth,
    minHeight,
    aspect: pageAspect ? windowAspect(containerAspect, pageAspect) : undefined,
  };
  const window = toCropWindow(bbox, opts);

  return (
    <div
      className={`relative overflow-hidden rounded-md border border-border bg-white ${className}`}
      style={{
        ...style,
        backgroundImage: `url(${imageUrl})`,
        backgroundRepeat: "no-repeat",
        ...toCropStyle(bbox, opts),
      }}
    >
      <div
        className="absolute rounded-[2px]"
        style={{
          ...toStyle(within(bbox, window)),
          border: `2px solid ${colour}`,
          backgroundColor: `${colour}1f`,
        }}
      />
    </div>
  );
}
