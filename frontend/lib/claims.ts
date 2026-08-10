import type { ClaimLabel } from "./api";

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
