import type { ClaimEvent, ClaimLabel, DoneEvent } from "./api";

/**
 * Explicit hex rather than Tailwind classes. The colour is chosen at runtime
 * from the claim index, and Tailwind only ships classes it can statically see,
 * so a computed class name would compile to nothing.
 *
 * ONE palette for both themes, chosen against white. These are drawn on the
 * page scan, which is white in dark mode too, so a set brightened for a dark
 * background would wash out on the only surface where the boxes have to be
 * readable. The cost is that the underline in the answer sits at about 1.9:1
 * against the dark card, well under 3:1, and that is acceptable only because
 * colour is a redundant channel here: every claim also carries its ordinal, in
 * the answer, in the vault and in the zoom, and the verdict is always spelled
 * out in words next to its icon. Nothing is knowable from hue alone.
 */
export const CLAIM_COLOURS = [
  "#1d4ed8",
  "#b91c4c",
  "#047857",
  "#b45309",
  "#6d28d9",
  "#0e7490",
  "#be123c",
  "#4d7c0f",
];

export function colourFor(index: number): string {
  return CLAIM_COLOURS[index % CLAIM_COLOURS.length];
}

export function labelText(label: string | null): string {
  return label ? label.replace(/_/g, " ") : "unverified";
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export type Tone = "ok" | "warn" | "danger" | "neutral";

/**
 * The verifier's four labels mapped to three tones.
 *
 * partially_supported and insufficient_evidence are both "warn" on purpose:
 * one says the evidence covers part of the claim, the other says it was not
 * found, and neither is a rejection. Only unsupported is a contradiction. The
 * label text is always shown next to the tone, because tone alone would put
 * meaning in colour.
 */
export function toneFor(label: ClaimLabel | null): Tone {
  if (label === "supported") return "ok";
  if (label === "unsupported") return "danger";
  if (label === null) return "neutral";
  return "warn";
}

/**
 * Split the shown claims into paragraphs at the breaks the reader marked.
 *
 * The break lives on the claim rather than being inferred from the text,
 * because only the reader knows where the topic turns.
 *
 * The first survivor always opens a paragraph regardless of its own flag.
 * That is not about avoiding an empty block: it is what stops the loop from
 * crashing. Verification removes claims after drafting, so the common case is
 * that NO survivor carries the flag at all; without the `groups.length === 0`
 * clause, the first claim would fall to the `else` branch and try to append
 * to `groups[-1]`, which is `undefined`, and `.push` on it throws inside
 * render.
 *
 * This cannot recover a break whose claim was withheld. When the claim
 * carrying the flag is gone, the break is genuinely lost, because it was the
 * only thing that knew where the topic turned, and the survivors merge into
 * one paragraph. That is the correct degradation, not a bug this function is
 * meant to fix.
 */
/**
 * Whether the system is declining to answer, as early as it is knowable.
 *
 * The rule itself is not restated here. `abstained_overall` and
 * `abstains_answer` are both computed server-side from `Claim.withheld` and
 * `LEAD_INDEX`; this only picks whichever of the two has arrived.
 *
 * It cannot DISAGREE with the server, only precede it. `abstains_answer` is
 * `index == LEAD_INDEX and withheld`, which strictly implies
 * `abstained_overall`, so a refusal shown on a claim frame is never one the
 * eventual `done` frame contradicts. It is the same conclusion, reached before
 * the remaining verifier calls are spent. Without it the answer panel renders
 * surviving support under an "Answer" heading for tens of seconds on a serial
 * GPU and then retracts the assertion that those sentences answer the
 * question, which is the one thing the lead rule was added to prevent.
 *
 * It reads the flag off whichever claim carries it rather than indexing
 * `claims[0]`, so it holds regardless of arrival order. Nothing emits claims
 * out of order today, and this stops that from being load-bearing.
 */
export function isAbstaining(claims: ClaimEvent[], done: DoneEvent | null): boolean {
  if (done) return done.abstained_overall;
  return claims.some((c) => c.abstains_answer);
}

export function groupIntoParagraphs(claims: ClaimEvent[]): ClaimEvent[][] {
  const groups: ClaimEvent[][] = [];
  for (const claim of claims) {
    if (groups.length === 0 || claim.starts_paragraph) groups.push([claim]);
    else groups[groups.length - 1].push(claim);
  }
  return groups;
}
