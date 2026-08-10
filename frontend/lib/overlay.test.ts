import { describe, expect, it } from "vitest";
import {
  toCropStyle,
  toCropWindow,
  toStyle,
  windowAspect,
  type BBox,
} from "./overlay";

describe("toStyle", () => {
  it("maps a normalized bbox to CSS percentages", () => {
    expect(toStyle([0.1, 0.2, 0.5, 0.6])).toEqual({
      left: "10%",
      top: "20%",
      width: "40%",
      height: "40%",
    });
  });

  // The bbox convention is (x0, y0, x1, y1) with origin top-left. S3 shipped a
  // patch-grid transposition that produced correctly-shaped, plausible output
  // and was wrong; the same bug is one swap away here.
  //
  // Moving ONE axis and asserting the other did not react is what discriminates.
  // The obvious version of this test, comparing toStyle(box) against
  // toStyle(swapped box), PASSES under a transposed implementation: the swap is
  // applied consistently to both inputs, so the two results still differ from
  // each other while both being wrong. Measured, not reasoned: a deliberately
  // transposed toStyle failed tests 1 and 4 and passed that version of this one.
  it("maps x0 to left and y0 to top independently", () => {
    const base = toStyle([0.1, 0.2, 0.5, 0.6]);
    const movedInX = toStyle([0.3, 0.2, 0.7, 0.6]);
    const movedInY = toStyle([0.1, 0.4, 0.5, 0.8]);

    expect(movedInX.top).toEqual(base.top);
    expect(movedInX.left).not.toEqual(base.left);
    expect(movedInY.left).toEqual(base.left);
    expect(movedInY.top).not.toEqual(base.top);
  });

  it("handles a full-page box", () => {
    expect(toStyle([0, 0, 1, 1])).toEqual({
      left: "0%",
      top: "0%",
      width: "100%",
      height: "100%",
    });
  });

  // A line box measured on proposal.pdf is 0.0142 tall. Rounding that to a
  // whole percent would collapse it to zero height and draw nothing.
  it("keeps sub-percent heights", () => {
    const style = toStyle([0.1, 0.5, 0.9, 0.5142]);
    expect(parseFloat(style.height)).toBeGreaterThan(0);
    expect(parseFloat(style.height)).toBeLessThan(2);
  });
});

describe("toCropStyle", () => {
  it("magnifies a window by the reciprocal of its size", () => {
    // Half the page in each axis, so the image is drawn at 200% and the visible
    // window is one quarter of it.
    const crop = toCropStyle([0.25, 0.25, 0.75, 0.75], { minWidth: 0, minHeight: 0 });

    expect(crop.backgroundSize).toBe("200.0000% 200.0000%");
    // Centred: the background has travelled half of its scrollable remainder.
    expect(crop.backgroundPosition).toBe("50.0000% 50.0000%");
  });

  // Same discriminating shape as the toStyle test above, and for the same
  // reason: a transposition applied consistently to two inputs still produces
  // two different-looking results. Move ONE axis, assert the other did not
  // react.
  it("maps x to the horizontal axis and y to the vertical independently", () => {
    const opts = { minWidth: 0, minHeight: 0 };
    const base = toCropStyle([0.2, 0.2, 0.4, 0.4], opts);
    const movedInX = toCropStyle([0.5, 0.2, 0.7, 0.4], opts);
    const movedInY = toCropStyle([0.2, 0.5, 0.4, 0.7], opts);

    const xOf = (c: { backgroundPosition: string }) => c.backgroundPosition.split(" ")[0];
    const yOf = (c: { backgroundPosition: string }) => c.backgroundPosition.split(" ")[1];

    expect(yOf(movedInX)).toEqual(yOf(base));
    expect(xOf(movedInX)).not.toEqual(xOf(base));
    expect(xOf(movedInY)).toEqual(xOf(base));
    expect(yOf(movedInY)).not.toEqual(yOf(base));
  });

  // THE test of this function. A real line box is 0.0142 tall, which cropped to
  // its own bounds magnifies seventy times and renders as unreadable
  // letterforms with no clue where on the page they came from.
  it("pads a line box out to a legible window", () => {
    const crop = toCropStyle([0.1, 0.5, 0.9, 0.5142]);
    const [, sy] = crop.backgroundSize.split(" ").map(parseFloat);

    expect(sy).toBeLessThan(2000);
    expect(sy).toBeCloseTo(100 / 0.06, 1);
  });

  it("keeps the window on the page at an edge", () => {
    // Centring on a box in the top-left corner would put the window off-page.
    const crop = toCropStyle([0, 0, 0.05, 0.05]);
    const [px, py] = crop.backgroundPosition.split(" ").map(parseFloat);

    expect(px).toBe(0);
    expect(py).toBe(0);
  });

  // NaN in a style string is dropped silently by the browser, so the crop would
  // render as an un-zoomed page rather than as a visible failure.
  it("produces a finite position for a full-page box", () => {
    const crop = toCropStyle([0, 0, 1, 1]);

    expect(crop.backgroundPosition).toBe("0.0000% 0.0000%");
    expect(crop.backgroundSize).toBe("100.0000% 100.0000%");
  });
});

describe("toCropWindow aspect correction", () => {
  // THE test of the aspect option. backgroundSize scales the two axes
  // independently, so a window whose ratio does not match its container is
  // drawn stretched. Measured before the fix: a line region padded to
  // 0.74 x 0.10 of a 1241x1754 page, shown in an 896x381 box, rendered 2.2
  // times too tall. A zoom that restretches the document it exists to show
  // faithfully is worse than no zoom.
  it("grows the deficient axis to reach the requested ratio", () => {
    const pageAspect = 1241 / 1754;
    const aspect = windowAspect(896 / 381, pageAspect);
    const [x0, y0, x1, y1] = toCropWindow([0.13, 0.3, 0.87, 0.315], { aspect });

    expect((x1 - x0) / (y1 - y0)).toBeCloseTo(aspect, 4);
  });

  it("keeps the region inside the window at the requested ratio", () => {
    const bbox: BBox = [0.13, 0.3, 0.87, 0.315];
    const [x0, y0, x1, y1] = toCropWindow(bbox, { aspect: 4 });

    expect(x0).toBeLessThanOrEqual(bbox[0] + 1e-9);
    expect(x1).toBeGreaterThanOrEqual(bbox[2] - 1e-9);
    expect(y0).toBeLessThanOrEqual(bbox[1] + 1e-9);
    expect(y1).toBeGreaterThanOrEqual(bbox[3] - 1e-9);
  });

  // No undistorted window can be 0.74 of the page wide and 1.48 of it tall,
  // because the page is not that tall. The honest outcome is the whole page
  // height and a letterboxed container, NOT a narrower window: an earlier
  // version scaled both axes to fit and cut 0.74 of page width down to 0.50,
  // slicing the region it was supposed to be showing in half.
  it("keeps the full region width when the ratio cannot be met", () => {
    const bbox: BBox = [0.13, 0.3, 0.87, 0.315];
    const [x0, y0, x1, y1] = toCropWindow(bbox, { aspect: 0.5 });

    expect(x0).toBeLessThanOrEqual(bbox[0] + 1e-9);
    expect(x1).toBeGreaterThanOrEqual(bbox[2] - 1e-9);
    expect(y1 - y0).toBeCloseTo(1, 6);
  });

  // The height is derived by the browser from the real image in this mode, so
  // no arithmetic here can stretch it.
  it("defers the height to the image when an aspect is known", () => {
    const aspect = windowAspect(896 / 381, 1241 / 1754);
    const crop = toCropStyle([0.13, 0.3, 0.87, 0.315], { aspect });

    expect(crop.backgroundSize.endsWith(" auto")).toBe(true);
    expect(crop.backgroundSize.startsWith("135.")).toBe(true);
  });
});
