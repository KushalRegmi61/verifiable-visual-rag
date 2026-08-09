"use client";

import { useCallback, useState } from "react";
import {
  API,
  ask,
  type Candidate,
  type ClaimEvent,
  type DoneEvent,
  type RetrievedEvent,
} from "@/lib/api";
import { toStyle } from "@/lib/overlay";

// Explicit hex values rather than Tailwind classes. The colour is chosen at
// runtime from the claim index, and Tailwind only ships classes it can see in
// the source, so a computed class name would compile to nothing.
const CLAIM_COLOURS = [
  "#2563eb",
  "#db2777",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#dc2626",
  "#4d7c0f",
];

function colourFor(index: number): string {
  return CLAIM_COLOURS[index % CLAIM_COLOURS.length];
}

function labelText(label: string | null): string {
  return label ? label.replace(/_/g, " ") : "unverified";
}

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [pending, setPending] = useState(false);
  const [retrieved, setRetrieved] = useState<RetrievedEvent | null>(null);
  // Held separately from `retrieved` and NOT cleared on a pinned re-ask. The
  // pinned branch of _choose_page returns candidates: [] by construction, so
  // reading the list off `retrieved` made it single-use: click one alternate
  // page and the row vanishes, and getting back to the original page means
  // retyping the question and paying for retrieval plus the whole
  // reader/verifier loop again. This is the UI's only affordance for the
  // retrieval-was-wrong case, which is exactly the case worth demonstrating.
  const [alternates, setAlternates] = useState<Candidate[]>([]);
  const [expected, setExpected] = useState<number | null>(null);
  const [claims, setClaims] = useState<ClaimEvent[]>([]);
  const [done, setDone] = useState<DoneEvent | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  const run = useCallback(async (q: string, pin?: { doc: string; page: number }) => {
    const text = q.trim();
    if (!text) return;
    setPending(true);
    setAsked(text);
    setError(null);
    setRetrieved(null);
    // Kept across a pinned re-ask, cleared when a new question is asked: the
    // ranking belongs to the question, not to the page currently displayed.
    if (!pin) setAlternates([]);
    setExpected(null);
    setClaims([]);
    setDone(null);
    setHovered(null);
    try {
      await ask(
        pin ? { question: text, doc: pin.doc, page: pin.page } : { question: text },
        {
          onRetrieved: (e) => {
            setRetrieved(e);
            // Only a fresh retrieval knows the ranking. A pinned re-ask carries
            // an empty list, and overwriting with it would discard the only
            // record of what else was considered.
            //
            // The top hit is prepended because the service sends candidates as
            // hits[1:]. Without it, clicking an alternate is one-way: there
            // would be no button for the page retrieval actually chose.
            if (!pin) {
              setAlternates([
                { doc_sha: e.doc_sha, page: e.page, score: e.score ?? 0 },
                ...e.candidates,
              ]);
            }
          },
          onClaims: setExpected,
          // Appended rather than replaced: every claim arrives as its own
          // event, already verified, and the list grows as they land.
          onClaim: (c) => setClaims((prev) => [...prev, c]),
          onDone: setDone,
          onError: setError,
        },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPending(false);
    }
  }, []);

  // The one line that guarantees a withheld claim can never reach the overlay.
  // The service already sends it with regions: [], and this makes the display
  // side agree rather than relying on that alone.
  const shown = claims.filter((c) => !c.withheld);
  const withheld = claims.filter((c) => c.withheld);
  const abstained = done?.abstained_overall ?? false;
  const pageImage = retrieved
    ? `${API}/documents/${retrieved.doc_sha}/pages/${retrieved.page}/image`
    : null;

  return (
    <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-semibold tracking-tight">Verifiable Visual RAG</h1>
        <p className="mt-1 text-sm text-black/60 dark:text-white/60">
          Every claim is checked by a second model and pinned to a region of the page it
          came from. Claims that fail the check are withheld, not shown.
        </p>
      </header>

      <form
        className="flex flex-col gap-2 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          void run(question);
        }}
      >
        <input
          className="flex-1 rounded-md border border-black/15 bg-white px-3 py-2 text-sm outline-none focus:border-black/40 dark:border-white/20 dark:bg-white/5 dark:focus:border-white/50"
          placeholder="Ask a question about an indexed document"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={pending}
        />
        <button
          type="submit"
          disabled={pending || !question.trim()}
          className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-40 dark:bg-white dark:text-black"
        >
          {pending ? "Working" : "Ask"}
        </button>
      </form>

      {error && (
        <div
          role="alert"
          className="mt-4 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-sm text-red-700 dark:text-red-300"
        >
          {error}
        </div>
      )}

      {asked && (
        <div className="mt-6 grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <section className="min-w-0">
            {retrieved?.warning && (
              <div className="mb-4 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-200">
                {retrieved.warning}
              </div>
            )}

            {abstained && (
              <div className="mb-4 rounded-xl border-2 border-amber-500/60 bg-amber-500/10 px-4 py-4">
                <p className="text-sm font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
                  No answer given
                </p>
                <p className="mt-1 text-sm text-black/70 dark:text-white/70">
                  The verifier rejected every claim the reader produced, so the system is
                  declining to answer this question rather than showing an answer it cannot
                  support.
                  {/* Without this, an unembedded page reads as a judgement about the
                      evidence when the real cause is that nobody ran `vvrag embed`. */}
                  {retrieved?.warning &&
                    " Note the warning above: this page has no embeddings, so that" +
                      " rejection is very likely a missing index rather than weak evidence."}
                </p>
              </div>
            )}

            {pending && expected === null && (
              <p className="text-sm text-black/50 dark:text-white/50">
                {retrieved ? "Reading the page" : "Retrieving a page"}
              </p>
            )}

            {expected !== null && claims.length < expected && (
              <p className="text-sm text-black/50 dark:text-white/50">
                Verifying claim {claims.length + 1} of {expected}
              </p>
            )}

            {shown.length > 0 && (
              <ol className="mt-3 space-y-3">
                {shown.map((c) => {
                  const colour = colourFor(c.index);
                  const dimmed = hovered !== null && hovered !== c.index;
                  return (
                    <li
                      key={c.index}
                      onMouseEnter={() => setHovered(c.index)}
                      onMouseLeave={() => setHovered(null)}
                      style={{ borderLeftColor: colour }}
                      className={`rounded-md border border-black/10 border-l-4 bg-black/[0.02] px-3 py-3 transition-opacity dark:border-white/10 dark:bg-white/[0.04] ${
                        dimmed ? "opacity-50" : "opacity-100"
                      }`}
                    >
                      <p className="text-sm leading-relaxed">{c.text}</p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                        <span
                          className="rounded-full px-2 py-0.5 font-medium text-white"
                          style={{ backgroundColor: colour }}
                        >
                          {labelText(c.label)}
                        </span>
                        <span className="text-black/50 dark:text-white/50">
                          confidence {pct(c.confidence)}
                        </span>
                        {c.compound && (
                          <span
                            className="rounded-full border border-black/20 px-2 py-0.5 text-black/60 dark:border-white/25 dark:text-white/60"
                            title="This claim asserts more than one thing, so a single region can only evidence part of it."
                          >
                            compound
                          </span>
                        )}
                        {c.regions.map((r, i) => (
                          <span
                            key={i}
                            className="rounded-full border border-black/20 px-2 py-0.5 text-black/60 dark:border-white/25 dark:text-white/60"
                            title={r.text ?? ""}
                          >
                            {r.modality}
                            {r.resolution ? ` / ${r.resolution}` : ""}
                          </span>
                        ))}
                      </div>
                      {c.reason && (
                        <p className="mt-2 text-xs text-black/50 dark:text-white/50">
                          {c.reason}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ol>
            )}

            {withheld.length > 0 && (
              <details className="mt-5 rounded-md border border-black/10 bg-black/[0.02] px-3 py-2 dark:border-white/10 dark:bg-white/[0.04]">
                <summary className="cursor-pointer text-sm font-medium">
                  {withheld.length} claim{withheld.length === 1 ? "" : "s"} withheld by the
                  verifier
                </summary>
                <ul className="mt-3 space-y-3">
                  {withheld.map((c) => (
                    <li
                      key={c.index}
                      className="rounded-md border border-dashed border-black/20 px-3 py-2 dark:border-white/25"
                    >
                      <p className="text-sm text-black/60 line-through decoration-black/30 dark:text-white/60 dark:decoration-white/30">
                        {c.text}
                      </p>
                      <p className="mt-1 text-xs text-black/50 dark:text-white/50">
                        {labelText(c.label)}, confidence {pct(c.confidence)}
                      </p>
                      <p className="mt-1 text-xs text-black/50 dark:text-white/50">
                        {c.reason ?? "The verifier gave no reason."}
                      </p>
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs text-black/40 dark:text-white/40">
                  Withheld claims carry no evidence region, so nothing is drawn for them on
                  the page.
                </p>
              </details>
            )}

            {done && !abstained && shown.length === 0 && (
              <p className="mt-3 text-sm text-black/50 dark:text-white/50">
                No claims to show.
              </p>
            )}
          </section>

          <section className="min-w-0">
            {retrieved ? (
              <>
                <div className="mb-2 flex flex-wrap items-baseline gap-2 text-xs text-black/60 dark:text-white/60">
                  <span className="font-medium text-black dark:text-white">
                    {retrieved.doc_name}
                  </span>
                  <span>page {retrieved.page}</span>
                  {retrieved.score !== null && <span>score {retrieved.score.toFixed(3)}</span>}
                </div>

                <div className="relative w-full overflow-hidden rounded-md border border-black/10 bg-white dark:border-white/15">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={pageImage!}
                    alt={`${retrieved.doc_name} page ${retrieved.page}`}
                    className="block w-full"
                  />
                  {shown.flatMap((c) =>
                    c.regions.map((r, i) => {
                      const colour = colourFor(c.index);
                      const dimmed = hovered !== null && hovered !== c.index;
                      return (
                        <div
                          key={`${c.index}-${i}`}
                          title={[
                            c.text,
                            c.reason ?? "",
                            r.resolution === "block"
                              ? "Block level: the heatmap could not separate the lines inside this block."
                              : "",
                          ]
                            .filter(Boolean)
                            .join("\n\n")}
                          style={{
                            ...toStyle(r.bbox),
                            position: "absolute",
                            borderColor: colour,
                            backgroundColor: `${colour}22`,
                            borderStyle: r.resolution === "block" ? "dashed" : "solid",
                            borderWidth: 2,
                            opacity: dimmed ? 0.25 : 1,
                          }}
                          className="pointer-events-auto rounded-[2px] transition-opacity"
                          onMouseEnter={() => setHovered(c.index)}
                          onMouseLeave={() => setHovered(null)}
                        />
                      );
                    }),
                  )}
                </div>

                <p className="mt-2 text-xs text-black/45 dark:text-white/45">
                  A dashed outline means the region stayed at block level because the
                  heatmap could not separate the lines inside it. Solid outlines are
                  line-level hits. Every box comes from the document text layer, never
                  drawn from the heatmap.
                </p>

                {alternates.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs font-medium text-black/60 dark:text-white/60">
                      Retrieved pages
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {alternates.map((cand: Candidate) => {
                        const current =
                          cand.doc_sha === retrieved.doc_sha && cand.page === retrieved.page;
                        return (
                          <button
                            key={`${cand.doc_sha}-${cand.page}`}
                            type="button"
                            disabled={pending || current}
                            onClick={() =>
                              void run(asked, { doc: cand.doc_sha, page: cand.page })
                            }
                            className="rounded-full border border-black/15 px-3 py-1 text-xs disabled:opacity-40 hover:border-black/40 dark:border-white/20 dark:hover:border-white/50"
                          >
                            page {cand.page}
                            <span className="ml-1 text-black/45 dark:text-white/45">
                              {cand.score.toFixed(3)}
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <p className="text-sm text-black/45 dark:text-white/45">
                The retrieved page appears here.
              </p>
            )}
          </section>
        </div>
      )}
    </main>
  );
}
