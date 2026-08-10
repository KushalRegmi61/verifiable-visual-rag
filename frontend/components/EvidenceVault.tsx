"use client";

import type { ClaimEvent } from "@/lib/api";
import { colourFor, labelText, pct, toneFor, type Tone } from "@/lib/claims";
import type { Region } from "@/lib/overlay";
import { AlertIcon, BlockedIcon, CheckIcon, ChevronIcon, ExpandIcon, VaultIcon } from "./icons";
import { RegionCrop } from "./RegionCrop";

const TONE_CLASS: Record<Tone, string> = {
  ok: "text-ok border-ok/35 bg-ok/8",
  warn: "text-warn border-warn/35 bg-warn/8",
  danger: "text-danger border-danger/35 bg-danger/8",
  neutral: "text-muted border-border bg-sunken",
};

function ToneIcon({ tone, className }: { tone: Tone; className?: string }) {
  if (tone === "ok") return <CheckIcon className={className} />;
  if (tone === "danger") return <BlockedIcon className={className} />;
  return <AlertIcon className={className} />;
}

type Props = {
  shown: ClaimEvent[];
  withheld: ClaimEvent[];
  /**
   * The page image for a given claim, resolved per claim rather than once.
   * A claim is grounded against every page the reader saw, so its evidence is
   * not always on the page retrieval ranked first, and a single URL here
   * cropped the WRONG page at the right coordinates: real ink, wrong page,
   * looking entirely correct. That is a fabricated citation in miniature.
   */
  imageUrlFor: (claim: ClaimEvent) => string | null;
  expanded: number | null;
  hovered: number | null;
  onToggle: (index: number) => void;
  onHover: (index: number | null) => void;
  onZoom: (claim: ClaimEvent, region: Region) => void;
  pageAspect: number | null;
};

const THUMB_ASPECT = 5 / 2;

/**
 * Every claim with its evidence, collapsed by default.
 *
 * Progressive disclosure rather than a flat list: a page routinely produces ten
 * claims, and rendering ten reasons, ten verdicts and ten crops at once buries
 * the answer the user came for under its own audit trail. Collapsed, a claim is
 * one line and a verdict. Opened, it is everything the system knows about why
 * that claim was allowed through.
 *
 * Withheld claims live at the bottom, behind their own disclosure, and carry no
 * geometry: the service strips a rejected claim's regions before they leave the
 * process, so there is nothing to draw and nothing to zoom into. That is the
 * guarantee, not a styling choice.
 */
export function EvidenceVault({
  shown,
  withheld,
  imageUrlFor,
  expanded,
  hovered,
  onToggle,
  onHover,
  onZoom,
  pageAspect,
}: Props) {
  if (shown.length === 0 && withheld.length === 0) return null;

  return (
    <section aria-labelledby="vault-heading" className="mt-6">
      <div className="flex items-center gap-2 px-1">
        <VaultIcon className="h-4 w-4 text-faint" />
        <h2
          id="vault-heading"
          className="text-[11px] font-semibold uppercase tracking-[0.14em] text-faint"
        >
          Evidence vault
        </h2>
        <span className="text-[11px] text-faint tnum">
          {shown.length} verified
          {withheld.length > 0 && `, ${withheld.length} withheld`}
        </span>
      </div>

      <ol className="mt-3 space-y-2">
        {shown.map((c) => {
          const colour = colourFor(c.index);
          const tone = toneFor(c.label);
          const isOpen = expanded === c.index;
          const dimmed = hovered !== null && hovered !== c.index && !isOpen;

          return (
            <li
              key={c.index}
              id={`claim-${c.index}`}
              onMouseEnter={() => onHover(c.index)}
              onMouseLeave={() => onHover(null)}
              style={{ borderLeftColor: colour }}
              className={`vv-rise overflow-hidden rounded-xl border border-l-[3px] border-border bg-surface transition-opacity duration-150 ${
                dimmed ? "opacity-55" : "opacity-100"
              }`}
            >
              <button
                type="button"
                onClick={() => onToggle(c.index)}
                aria-expanded={isOpen}
                aria-controls={`claim-body-${c.index}`}
                className="flex w-full cursor-pointer items-start gap-3 p-3 text-left transition-colors duration-150 hover:bg-sunken"
              >
                <span
                  aria-hidden
                  className="mt-px inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold text-white tnum"
                  style={{ backgroundColor: colour }}
                >
                  {c.index + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block text-sm leading-relaxed">{c.text}</span>
                  <span className="mt-1.5 flex flex-wrap items-center gap-1.5">
                    {/* Icon and text, never tone alone: the four verdicts must
                        stay distinguishable without colour vision. */}
                    <span
                      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium ${TONE_CLASS[tone]}`}
                    >
                      <ToneIcon tone={tone} className="h-3 w-3" />
                      {labelText(c.label)}
                    </span>
                    <span className="text-[11px] text-faint tnum">
                      {pct(c.confidence)} confident
                    </span>
                    {c.compound && (
                      <span
                        className="rounded-full border border-border px-2 py-0.5 text-[11px] text-muted"
                        title="This claim asserts more than one thing, so a single region can only evidence part of it."
                      >
                        compound
                      </span>
                    )}
                  </span>
                </span>
                <ChevronIcon
                  className={`mt-1 h-4 w-4 shrink-0 text-faint transition-transform duration-200 ${
                    isOpen ? "rotate-90" : ""
                  }`}
                />
              </button>

              {isOpen && (
                <div
                  id={`claim-body-${c.index}`}
                  className="border-t border-border bg-sunken/60 px-3 py-3"
                >
                  {imageUrlFor(c) && c.regions.length > 0 && (
                    <div className="grid gap-2 sm:grid-cols-2">
                      {c.regions.map((r, i) => (
                        <button
                          key={i}
                          type="button"
                          onClick={() => onZoom(c, r)}
                          aria-label={`Zoom into evidence region ${i + 1} for claim ${c.index + 1}`}
                          className="group relative block cursor-pointer overflow-hidden rounded-md text-left"
                        >
                          <RegionCrop
                            imageUrl={imageUrlFor(c)!}
                            bbox={r.bbox}
                            colour={colour}
                            containerAspect={THUMB_ASPECT}
                            pageAspect={pageAspect}
                            className="w-full transition-transform duration-200 group-hover:scale-[1.02]"
                            style={{ aspectRatio: THUMB_ASPECT }}
                          />
                          <span className="pointer-events-none absolute right-1.5 top-1.5 inline-flex h-6 w-6 items-center justify-center rounded-md bg-black/55 text-white opacity-0 transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100">
                            <ExpandIcon className="h-3.5 w-3.5" />
                          </span>
                          <span className="mt-1 block text-[11px] text-faint">
                            {r.modality}
                            {r.resolution ? ` / ${r.resolution}` : ""}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                  {c.reason && (
                    <p className="mt-3 text-xs leading-relaxed text-muted">
                      <span className="font-medium text-foreground">Verifier: </span>
                      {c.reason}
                    </p>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ol>

      {withheld.length > 0 && (
        <details className="mt-3 rounded-xl border border-dashed border-border-strong bg-surface/60">
          <summary className="cursor-pointer list-none p-3 text-sm font-medium">
            <span className="inline-flex items-center gap-2">
              <BlockedIcon className="h-4 w-4 text-faint" />
              {withheld.length} claim{withheld.length === 1 ? "" : "s"} withheld by the
              verifier
            </span>
          </summary>
          <ul className="space-y-2 px-3 pb-3">
            {withheld.map((c) => (
              <li key={c.index} className="rounded-lg border border-border bg-sunken/60 p-3">
                <p className="text-sm leading-relaxed text-muted line-through decoration-faint/60">
                  {c.text}
                </p>
                <p className="mt-1.5 text-[11px] text-faint">
                  {labelText(c.label)}, {pct(c.confidence)} confident
                </p>
                <p className="mt-1 text-xs leading-relaxed text-muted">
                  {c.reason ?? "The verifier gave no reason."}
                </p>
              </li>
            ))}
          </ul>
          <p className="border-t border-border px-3 py-2 text-[11px] text-faint">
            A withheld claim leaves the service with its regions already removed, so
            there is nothing here to draw on the page.
          </p>
        </details>
      )}
    </section>
  );
}
