import { describe, expect, it } from "vitest";
import { groupIntoParagraphs, isAbstaining } from "./claims";
import type { ClaimEvent, DoneEvent } from "./api";

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
    abstains_answer: false,
  };
}

/** A withheld lead, as the wire sends it: no regions, abstains_answer set. */
function abstainingLead(index = 0): ClaimEvent {
  return {
    ...claim(index),
    label: "unsupported",
    withheld: true,
    abstains_answer: true,
  };
}

function done(abstained_overall: boolean): DoneEvent {
  return { shown: 0, withheld: 0, abstained_overall };
}

describe("groupIntoParagraphs", () => {
  // THE test of this function. The common case in production is that no
  // survivor carries starts_paragraph at all, because the claim that had it
  // was withheld by verification. Without the `groups.length === 0` clause,
  // the first claim falls to the `else` branch and appends to `groups[-1]`,
  // which is `undefined`; this goes red with a TypeError thrown inside
  // render, not a failed assertion, and in a client component that takes
  // down the whole page.
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

  // Not a test of the empty-groups guard: claim 0 carries the flag itself, so
  // starts_paragraph alone opens the first group and `groups.length === 0` is
  // never reached. What this catches is a different, genuinely plausible
  // wrong implementation that seeds `groups` with one empty array up front
  // and pushes a new empty array on every flagged claim instead of pushing
  // the claim itself:
  //   const groups = [[]];
  //   for (const claim of claims) {
  //     if (claim.starts_paragraph) groups.push([]);
  //     groups[groups.length - 1].push(claim);
  //   }
  // That version passes every other test in this file and fails only this
  // one, with "expected 2 to be 1", because it opens a second, empty-until-
  // pushed-into group for a flag on the very first claim.
  it("does not open with an empty paragraph when the first claim carries the break", () => {
    const groups = groupIntoParagraphs([claim(0, true), claim(1)]);
    expect(groups.length).toBe(1);
    expect(groups.every((g) => g.length > 0)).toBe(true);
  });

  // The real production shape: the reader marked the break on claim 3, the
  // verifier withheld claim 3, and the browser receives claims 1, 2, 4, 5
  // with no break on any survivor and gaps in the indices (`shown` is
  // `claims.filter(c => !c.withheld)`, so gaps are the normal case, not an
  // edge case). The break is genuinely LOST here, and that is correct: the
  // claim that knew where the topic turned is off screen, so the survivors
  // merge into one paragraph. What must not happen is a crash.
  it("keeps one paragraph when the claim carrying the break was withheld", () => {
    const groups = groupIntoParagraphs([claim(1), claim(2), claim(4), claim(5)]);
    expect(groups.map((g) => g.map((c) => c.index))).toEqual([[1, 2, 4, 5]]);
  });

  it("returns nothing for an empty answer", () => {
    expect(groupIntoParagraphs([])).toEqual([]);
  });
});

describe("isAbstaining", () => {
  it("is false before any claim has arrived", () => {
    // Nothing has been judged yet, so there is nothing to decline. Returning
    // true here would make every question open on a refusal.
    expect(isAbstaining([], null)).toBe(false);
  });

  it("abstains on the lead's own claim frame, before done arrives", () => {
    // THE test. The server emits every ClaimVerified before AnswerComplete, so
    // waiting for `done` means rendering the surviving support under an
    // "Answer" heading for one verifier call per remaining claim and then
    // retracting it.
    expect(isAbstaining([abstainingLead()], null)).toBe(true);
  });

  it("does not abstain while the lead is surviving", () => {
    expect(isAbstaining([claim(0), claim(1)], null)).toBe(false);
  });

  it("does not abstain when only a later claim was withheld", () => {
    // A withheld claim that is not the lead carries abstains_answer false, so
    // support failing verification never turns an answer into a refusal.
    const support: ClaimEvent = { ...claim(2), withheld: true };
    expect(isAbstaining([claim(0), support], null)).toBe(false);
  });

  it("takes done as the authority once it arrives", () => {
    // done abstains on a second condition the claim frames cannot express,
    // "nothing survived", which is reachable with an empty list on this side.
    expect(isAbstaining([], done(true))).toBe(true);
  });

  it("prefers done over the claim frames when both are present", () => {
    expect(isAbstaining([claim(0)], done(true))).toBe(true);
    expect(isAbstaining([abstainingLead()], done(false))).toBe(false);
  });

  it("finds the flag regardless of arrival order", () => {
    // It reads the flag off whichever claim carries it rather than indexing
    // claims[0], so nothing here depends on the lead landing first.
    expect(isAbstaining([claim(1), claim(2), abstainingLead(0)], null)).toBe(true);
  });
});
