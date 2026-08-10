"use client";

import { useEffect, useRef } from "react";
import type { ClaimEvent } from "@/lib/api";
import { colourFor, labelText, pct } from "@/lib/claims";
import type { Region } from "@/lib/overlay";
import { CloseIcon } from "./icons";
import { RegionCrop } from "./RegionCrop";

type Props = {
  imageUrl: string;
  claim: ClaimEvent;
  region: Region;
  docName: string;
  page: number;
  pageAspect: number | null;
  onClose: () => void;
};

// The crop container is a fixed ratio rather than a viewport height, so the
// window it implies is a constant the crop maths can be given up front.
const ZOOM_ASPECT = 16 / 7;

/**
 * The zoomed look at one region, opened by clicking its box or its crop.
 *
 * Deliberately shows the claim, the verifier's verdict, and the text the
 * grounder actually matched all in one frame. Zooming into the pixels alone
 * would answer "what does this say" while leaving "and is this why the claim
 * was accepted" unanswered, and the second question is the one this project
 * exists to make answerable.
 */
export function ZoomOverlay({
  imageUrl,
  claim,
  region,
  docName,
  page,
  pageAspect,
  onClose,
}: Props) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    // Focus moves into the dialog so the next Tab stays inside it, and Escape
    // is the escape route every modal owes the user.
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    // The page behind must not scroll while a full-screen layer is over it.
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const colour = colourFor(claim.index);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm sm:p-8"
      role="dialog"
      aria-modal="true"
      aria-label={`Evidence for claim ${claim.index + 1}`}
      onClick={onClose}
    >
      <div
        className="vv-rise w-full max-w-4xl overflow-hidden rounded-2xl border border-border bg-surface shadow-2xl"
        // The backdrop closes on click, so a click that lands on the panel must
        // not bubble up to it.
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="min-w-0">
            <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-faint">
              {docName}, page {page}
            </p>
            <p className="mt-1 flex items-center gap-2 text-sm font-medium">
              <span
                className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold text-white tnum"
                style={{ backgroundColor: colour }}
              >
                {claim.index + 1}
              </span>
              <span className="truncate">{claim.text}</span>
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close evidence view"
            className="-m-2 shrink-0 cursor-pointer rounded-lg p-2 text-muted transition-colors duration-150 hover:bg-sunken hover:text-foreground"
          >
            <CloseIcon className="h-5 w-5" />
          </button>
        </div>

        <RegionCrop
          imageUrl={imageUrl}
          bbox={region.bbox}
          colour={colour}
          // Wide enough to hold a whole text column. A window narrower than the
          // column cuts the surrounding lines mid-word, which reads as a
          // rendering fault rather than as a deliberate magnification.
          minWidth={0.66}
          minHeight={0.1}
          containerAspect={ZOOM_ASPECT}
          pageAspect={pageAspect}
          className="w-full rounded-none border-x-0"
          style={{ aspectRatio: ZOOM_ASPECT }}
        />

        <div className="grid gap-4 px-5 py-4 sm:grid-cols-[1fr_auto] sm:items-start">
          <div className="min-w-0">
            {region.text && (
              <>
                <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-faint">
                  Text layer at this region
                </p>
                {/* Monospace because this is the exact string the grounder
                    matched, not prose. Reading it as a quotation is the point. */}
                <p className="mt-1 font-mono text-xs leading-relaxed text-muted">
                  {region.text}
                </p>
              </>
            )}
            {claim.reason && (
              <>
                <p className="mt-3 text-[11px] font-medium uppercase tracking-[0.08em] text-faint">
                  Verifier
                </p>
                <p className="mt-1 text-sm leading-relaxed text-muted">{claim.reason}</p>
              </>
            )}
          </div>
          <dl className="flex shrink-0 gap-4 text-xs sm:flex-col sm:gap-2 sm:text-right">
            <div>
              <dt className="text-faint">Verdict</dt>
              <dd className="font-medium">{labelText(claim.label)}</dd>
            </div>
            <div>
              <dt className="text-faint">Confidence</dt>
              <dd className="font-medium tnum">{pct(claim.confidence)}</dd>
            </div>
            <div>
              <dt className="text-faint">Resolution</dt>
              <dd className="font-medium">
                {region.modality}
                {region.resolution ? ` / ${region.resolution}` : ""}
              </dd>
            </div>
          </dl>
        </div>
      </div>
    </div>
  );
}
