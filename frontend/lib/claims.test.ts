import { describe, expect, it } from "vitest";
import { groupIntoParagraphs } from "./claims";
import type { ClaimEvent } from "./api";

function claim(index: number, starts_paragraph = false): ClaimEvent {
  return {
    index,
    text: `claim ${index}`,
    label: "supported",
    confidence: 0.9,
    reason: null,
    compound: false,
    withheld: false,
    starts_paragraph,
    regions: [],
  };
}

describe("groupIntoParagraphs", () => {
  it("keeps an unbroken answer as one paragraph", () => {
    const groups = groupIntoParagraphs([claim(0), claim(1), claim(2)]);
    expect(groups.length).toBe(1);
    expect(groups[0].map((c) => c.index)).toEqual([0, 1, 2]);
  });

  it("breaks where a claim says it starts a paragraph", () => {
    const groups = groupIntoParagraphs([claim(0), claim(1), claim(2, true), claim(3)]);
    expect(groups.map((g) => g.map((c) => c.index))).toEqual([
      [0, 1],
      [2, 3],
    ]);
  });

  // THE test of this function. Verification removes claims, so the claim
  // carrying the break is frequently the one withheld. Starting a paragraph on
  // the FIRST survivor regardless keeps the answer from opening with an empty
  // block, and dropping the flag entirely would silently merge two topics.
  it("never produces an empty paragraph when the breaking claim is gone", () => {
    const groups = groupIntoParagraphs([claim(0, true), claim(1)]);
    expect(groups.length).toBe(1);
    expect(groups.every((g) => g.length > 0)).toBe(true);
  });

  it("returns nothing for an empty answer", () => {
    expect(groupIntoParagraphs([])).toEqual([]);
  });
});
