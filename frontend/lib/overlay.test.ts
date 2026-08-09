import { describe, expect, it } from "vitest";
import { toStyle } from "./overlay";

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
