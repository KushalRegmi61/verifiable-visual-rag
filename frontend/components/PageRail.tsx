"use client";

import type { Candidate } from "@/lib/api";

type Props = {
  candidates: Candidate[];
  activeDoc: string;
  activePage: number;
  isRead: (docSha: string, page: number) => boolean;
  pending: boolean;
  onSelect: (candidate: Candidate) => void;
};

/**
 * The other pages retrieval returned, ranked.
 *
 * A page already read is solid and comes back instantly; one not yet read is
 * dashed and costs a reader call plus a verifier call per claim on a GPU that
 * serves one request at a time. Showing the difference before the click is the
 * whole reason the two styles exist: the alternative is a row of identical
 * chips where one is free and the next takes a minute.
 *
 * The document name appears only when a candidate belongs to a DIFFERENT
 * document than the one on screen. Retrieval is corpus-wide and takes no
 * document filter, so a chip reading only "page 24" is indistinguishable from
 * page 24 of the document being displayed.
 */
export function PageRail({
  candidates,
  activeDoc,
  activePage,
  isRead,
  pending,
  onSelect,
}: Props) {
  if (candidates.length === 0) return null;

  return (
    <section aria-labelledby="rail-heading" className="mt-4">
      <h2
        id="rail-heading"
        className="px-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-faint"
      >
        Retrieved pages
      </h2>
      <div className="mt-2 flex flex-wrap gap-2">
        {candidates.map((cand) => {
          const isActive = cand.doc_sha === activeDoc && cand.page === activePage;
          const read = isRead(cand.doc_sha, cand.page);
          const elsewhere = cand.doc_sha !== activeDoc;

          return (
            <button
              key={`${cand.doc_sha}-${cand.page}`}
              type="button"
              disabled={pending || isActive}
              aria-current={isActive ? "true" : undefined}
              title={`${cand.doc_name} page ${cand.page}${
                read ? ", already read" : ", not read yet"
              }`}
              onClick={() => onSelect(cand)}
              className={`inline-flex min-h-[36px] cursor-pointer items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs transition-colors duration-150 disabled:cursor-default ${
                isActive
                  ? "border-accent bg-accent text-white"
                  : read
                    ? "border-border-strong bg-surface hover:border-accent hover:text-accent"
                    : "border-dashed border-border bg-transparent text-muted hover:border-accent hover:text-accent"
              } ${pending && !isActive ? "opacity-40" : ""}`}
            >
              {elsewhere && (
                <span className={isActive ? "opacity-75" : "text-faint"}>
                  {cand.doc_name}
                </span>
              )}
              <span className="tnum">page {cand.page}</span>
              <span className={`tnum ${isActive ? "opacity-75" : "text-faint"}`}>
                {cand.score.toFixed(3)}
              </span>
            </button>
          );
        })}
      </div>
      <p className="mt-2 px-1 text-[11px] leading-relaxed text-faint">
        Ranked by MaxSim over the page patch embeddings. A solid chip has already
        been read and returns instantly with its own claims and boxes; a dashed one
        costs a reader call plus a verifier call per claim.
      </p>
    </section>
  );
}
