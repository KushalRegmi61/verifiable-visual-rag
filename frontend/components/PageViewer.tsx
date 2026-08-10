"use client";

import type { ClaimEvent, RetrievedEvent } from "@/lib/api";
import { colourFor } from "@/lib/claims";
import { toStyle, type Region } from "@/lib/overlay";
import { PageIcon } from "./icons";

type Props = {
  retrieved: RetrievedEvent;
  imageUrl: string;
  /**
   * The page this image is of, which is not always `retrieved.page`: the
   * viewer follows the hovered or selected claim to whichever of the pages the
   * reader saw its evidence is on. Passed in rather than derived, because the
   * component draws only the regions belonging to it and the decision of which
   * page that is lives in `lib/claims` where it can be tested.
   */
  page: number;
  /** Null unless the page on screen is the one retrieval ranked. */
  score: number | null;
  shown: ClaimEvent[];
  hovered: number | null;
  onHover: (index: number | null) => void;
  onZoom: (claim: ClaimEvent, region: Region) => void;
  /**
   * Reports the page's intrinsic width/height once the image loads. The
   * retrieval payload carries no page dimensions, and the crop views need the
   * real ratio or they magnify a distorted page.
   */
  onMeasure: (aspect: number) => void;
};

/**
 * The page, with the regions drawn on it.
 *
 * ALIGNMENT. Every box is a percentage of its containing block, so the
 * container has to be exactly the rendered image box and nothing more. The
 * wrapper is w-fit around an image that is height-constrained and width-auto,
 * which makes the two rects identical at every viewport size. The previous
 * version stretched the image to the column width, so a portrait page ran far
 * past the fold and the boxes were correct only because the container happened
 * to match; constraining by height without w-fit would have left the container
 * wider than the image and slid every box left.
 */
export function PageViewer({
  retrieved,
  imageUrl,
  page,
  score,
  shown,
  hovered,
  onHover,
  onZoom,
  onMeasure,
}: Props) {
  return (
    <div>
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 px-1 text-xs">
        <span className="inline-flex items-center gap-1.5 font-medium">
          <PageIcon className="h-3.5 w-3.5 text-faint" />
          {retrieved.doc_name}
        </span>
        {/* `page`, not `retrieved.page`. Following a claim to another page
            while the header still names the retrieved one would caption the
            image with the wrong page number, which is worse than not
            captioning it: the caption is what a reader would quote. */}
        <span className="text-muted tnum">page {page}</span>
        {score !== null && (
          <span className="rounded-full bg-accent-soft px-2 py-0.5 font-medium text-accent tnum">
            {score.toFixed(3)}
          </span>
        )}
      </div>

      {/* The mat hugs the page rather than filling the column. A portrait scan
          is much narrower than the space beside the evidence vault, and a
          full-width mat around a narrow page reads as a rendering mistake. */}
      <div className="mx-auto w-fit max-w-full rounded-2xl border border-border bg-sunken p-3">
        <div className="relative w-fit max-w-full">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageUrl}
            alt={`${retrieved.doc_name} page ${retrieved.page}`}
            onLoad={(e) => {
              const img = e.currentTarget;
              if (img.naturalHeight > 0) onMeasure(img.naturalWidth / img.naturalHeight);
            }}
            className="block max-h-[calc(100vh-14rem)] w-auto max-w-full rounded-lg shadow-sm"
          />
          {/* Only the regions of the page actually on screen. A claim grounded
              on another of the pages the reader saw carries that page's
              coordinates, and drawing them here would put a box over unrelated
              text at the same position, which is a fabricated citation rather
              than a misplaced one. */}
          {shown.flatMap((c) =>
            c.regions
              .filter((r) => r.page === page)
              .map((r, i) => {
                const colour = colourFor(c.index);
                const dimmed = hovered !== null && hovered !== c.index;
                return (
                  <button
                    key={`${c.index}-${r.page}-${i}`}
                    type="button"
                    onClick={() => onZoom(c, r)}
                    onMouseEnter={() => onHover(c.index)}
                    onMouseLeave={() => onHover(null)}
                    aria-label={`Claim ${c.index + 1}: ${c.text}. Open zoomed evidence.`}
                    style={{
                      ...toStyle(r.bbox),
                      position: "absolute",
                      borderColor: colour,
                      backgroundColor: `${colour}${dimmed ? "10" : "26"}`,
                      // Dashed means the region stayed at block level, so the
                      // heatmap could not separate the lines inside it. A
                      // confident line hit and a coarse fallback must not look
                      // the same.
                      borderStyle: r.resolution === "block" ? "dashed" : "solid",
                      borderWidth: 2,
                      opacity: dimmed ? 0.3 : 1,
                    }}
                    className="cursor-pointer rounded-[2px] transition-all duration-150 hover:shadow-[0_0_0_3px_rgba(0,0,0,0.08)]"
                  />
                );
              }),
          )}
        </div>
      </div>

      <p className="mt-2 px-1 text-xs leading-relaxed text-faint">
        Click a box to zoom into it. A dashed outline means the region stayed at
        block level because the heatmap could not separate the lines inside it;
        solid outlines are line-level hits. Every box comes from the document text
        layer, never drawn from the heatmap.
      </p>
    </div>
  );
}
