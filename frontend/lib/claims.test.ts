import { describe, expect, it } from "vitest";
import { groupIntoParagraphs, isAbstaining, pageForClaim, type PageRef } from "./claims";
import type { ClaimEvent, DoneEvent } from "./api";
import type { Region } from "./overlay";

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

describe("pageForClaim", () => {
  const DOC = "a".repeat(64);
  // As the wire sends it: retrieval order, pages[0] the top page.
  const PAGES: PageRef[] = [
    { doc_sha: DOC, page: 4 },
    { doc_sha: DOC, page: 9 },
    { doc_sha: DOC, page: 2 },
  ];
  const TOP = PAGES[0];

  function region(page: number): Region {
    return {
      page,
      bbox: [0.1, 0.2, 0.5, 0.24],
      score: 12.5,
      modality: "visual",
      resolution: "line",
      text: null,
    };
  }

  function grounded(...pages: number[]): { regions: Region[] } {
    return { regions: pages.map(region) };
  }

  it("follows a claim to the page its evidence is on", () => {
    // The whole point: the evidence for this sentence is not on the page that
    // was retrieved, and the viewer has to move to it or the boxes it draws
    // belong to a page nobody is looking at.
    expect(pageForClaim(grounded(9), PAGES, TOP)).toEqual({ doc_sha: DOC, page: 9 });
  });

  it("stays on the page shown when the claim has no regions", () => {
    // THE test. A withheld claim ships with regions: [] by design, and so does
    // a claim whose citation was filtered out. Returning anything page-shaped
    // but empty here blanks the viewer, which reads as a broken page rather
    // than as "this claim has no evidence".
    const shown: PageRef = { doc_sha: DOC, page: 2 };
    expect(pageForClaim(grounded(), PAGES, shown)).toBe(shown);
  });

  it("stays on the page shown when the regions disagree about their page", () => {
    // Unreachable today: _best_region keeps the best region of one page, so a
    // claim's regions are always on one page. Pinned anyway because the
    // tempting implementation, regions[0].page, would answer 4 here and
    // present one page as THE page for evidence spread over two, with nothing
    // on screen to say otherwise.
    const shown: PageRef = { doc_sha: DOC, page: 4 };
    expect(pageForClaim(grounded(4, 9), PAGES, shown)).toBe(shown);
  });

  it("treats several regions on one page as that page", () => {
    // The disagreement guard must not catch the ordinary multi-region claim.
    expect(pageForClaim(grounded(9, 9, 9), PAGES, TOP)).toEqual({ doc_sha: DOC, page: 9 });
  });

  it("stays on the page shown when the claim's page was never retrieved", () => {
    // Defensive. `pages` is the only thing that knows a page's doc_sha, so
    // without an entry there is no honest URL to build; assuming the doc on
    // screen would fetch a real image of the wrong document's page 11.
    expect(pageForClaim(grounded(11), PAGES, TOP)).toBe(TOP);
  });

  it("does not assume the claim's page is the top one", () => {
    // Guards against returning pages[0] instead of the matching entry, which
    // passes every test above whose claim happens to sit on the top page.
    expect(pageForClaim(grounded(2), PAGES, TOP).page).toBe(2);
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
