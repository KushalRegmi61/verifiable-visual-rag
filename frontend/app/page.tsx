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
import type { Region } from "@/lib/overlay";
import { AnswerPanel } from "@/components/AnswerPanel";
import { EvidenceVault } from "@/components/EvidenceVault";
import { PageRail } from "@/components/PageRail";
import { PageViewer } from "@/components/PageViewer";
import { ZoomOverlay } from "@/components/ZoomOverlay";
import { AlertIcon, SearchIcon } from "@/components/icons";

/** Everything one page's answer is made of. */
type PageResult = {
  retrieved: RetrievedEvent;
  expected: number | null;
  claims: ClaimEvent[];
  done: DoneEvent | null;
};

type Zoom = { claim: ClaimEvent; region: Region };

function keyOf(docSha: string, page: number): string {
  return `${docSha}:${page}`;
}

export default function Home() {
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState("");
  const [pending, setPending] = useState(false);
  // Every page read for the current question, keyed doc_sha:page. Answering one
  // page costs a reader call plus a verifier call per claim on a serial GPU, so
  // re-reading a page the user has already seen just to get back to it is the
  // most expensive thing this UI could do. Keeping the results makes flipping
  // between retrieved pages free, and makes comparing what page 7 supports
  // against what page 12 supports possible at all, which is the only reason to
  // show more than one retrieved page.
  const [results, setResults] = useState<Record<string, PageResult>>({});
  const [active, setActive] = useState<string | null>(null);
  // Held separately from `results` and NOT cleared on a pinned re-ask. The
  // pinned branch of _choose_page returns candidates: [] by construction, so
  // reading the list off the current page made it single-use: click one
  // alternate page and the row vanishes, and getting back to the original page
  // means retyping the question. This is the UI's only affordance for the
  // retrieval-was-wrong case, which is exactly the case worth demonstrating.
  const [alternates, setAlternates] = useState<Candidate[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [zoom, setZoom] = useState<Zoom | null>(null);
  // Measured from the rendered page image rather than assumed. Reset on a page
  // change so a landscape page cannot be cropped using the ratio of the
  // portrait one before it, which would distort every crop until the new image
  // finished loading.
  const [pageAspect, setPageAspect] = useState<number | null>(null);

  const run = useCallback(
    async (q: string, pin?: { doc: string; page: number }) => {
      const text = q.trim();
      if (!text) return;

      // Already read for this question: show it, spend nothing. Only reachable
      // from a candidate chip, which is always pinned, so a fresh question can
      // never be short-circuited by a page left over from the last one.
      if (pin && results[keyOf(pin.doc, pin.page)]) {
        setActive(keyOf(pin.doc, pin.page));
        setHovered(null);
        setExpanded(null);
        setPageAspect(null);
        return;
      }

      setPending(true);
      setAsked(text);
      setError(null);
      setHovered(null);
      setExpanded(null);
      setZoom(null);
      setActive(null);
      setPageAspect(null);
      // Both the cache and the ranking belong to the question rather than to
      // the page on screen, so a new question drops both and a pinned re-ask
      // keeps both.
      if (!pin) {
        setResults({});
        setAlternates([]);
      }

      // Set by the retrieved frame, which always arrives first, and read by
      // every frame after it. A local rather than state, because the claim
      // handler needs the value inside this same stream.
      let key: string | null = null;
      const patch = (f: (r: PageResult) => PageResult) => {
        if (!key) return;
        const k = key;
        setResults((prev) => (prev[k] ? { ...prev, [k]: f(prev[k]) } : prev));
      };

      try {
        await ask(
          pin ? { question: text, doc: pin.doc, page: pin.page } : { question: text },
          {
            onRetrieved: (e) => {
              key = keyOf(e.doc_sha, e.page);
              setResults((prev) => ({
                ...prev,
                [key!]: { retrieved: e, expected: null, claims: [], done: null },
              }));
              setActive(key);
              // Only a fresh retrieval knows the ranking. A pinned re-ask
              // carries an empty list, and overwriting with it would discard
              // the only record of what else was considered.
              //
              // The top hit is prepended because the service sends candidates
              // as hits[1:]. Without it, clicking an alternate is one-way:
              // there would be no button for the page retrieval actually chose.
              if (!pin) {
                setAlternates([
                  {
                    doc_sha: e.doc_sha,
                    page: e.page,
                    score: e.score ?? 0,
                    doc_name: e.doc_name,
                  },
                  ...e.candidates,
                ]);
              }
            },
            onClaims: (n) => patch((r) => ({ ...r, expected: n })),
            // Appended rather than replaced: every claim arrives as its own
            // event, already verified, and the list grows as they land.
            onClaim: (c) => patch((r) => ({ ...r, claims: [...r.claims, c] })),
            onDone: (d) => patch((r) => ({ ...r, done: d })),
            onError: setError,
          },
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setPending(false);
      }
    },
    [results],
  );

  const currentPage = active ? (results[active] ?? null) : null;
  const retrieved = currentPage?.retrieved ?? null;
  const claims = currentPage?.claims ?? [];
  const expectedCount = currentPage?.expected ?? null;
  const done = currentPage?.done ?? null;

  // The one line that guarantees a withheld claim can never reach the overlay.
  // The service already sends it with regions: [], and this makes the display
  // side agree rather than relying on that alone.
  const shown = claims.filter((c) => !c.withheld);
  const withheld = claims.filter((c) => c.withheld);
  const abstained = done?.abstained_overall ?? false;
  const imageUrl = retrieved
    ? `${API}/documents/${retrieved.doc_sha}/pages/${retrieved.page}/image`
    : null;
  // A pinned re-ask does no retrieval, so its Retrieved event carries score:
  // null by construction. Reading the score off the ranking instead means the
  // header shows one for every page, which is what makes two pages comparable:
  // without it, switching to an alternate silently drops the only number that
  // says how well that page matched.
  const rankScore =
    retrieved?.score ??
    alternates.find(
      (c) => c.doc_sha === retrieved?.doc_sha && c.page === retrieved?.page,
    )?.score ??
    null;

  // Clicking a sentence in the answer opens its evidence and brings it into
  // view. Without the scroll the disclosure opens somewhere below the fold and
  // the click reads as having done nothing.
  const revealClaim = useCallback((index: number) => {
    setExpanded((prev) => (prev === index ? null : index));
    requestAnimationFrame(() => {
      document
        .getElementById(`claim-${index}`)
        ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
  }, []);

  const status = pending
    ? retrieved === null
      ? "Retrieving the best page"
      : expectedCount === null
        ? "Reading the page"
        : `Verifying claim ${Math.min(claims.length + 1, expectedCount)} of ${expectedCount}`
    : null;

  return (
    <>
      <header className="sticky top-0 z-30 border-b border-border bg-background/85 backdrop-blur-md">
        <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-3 px-5 py-3 lg:flex-row lg:items-center lg:gap-6 lg:px-8">
          <div className="shrink-0">
            <h1 className="text-sm font-semibold tracking-tight">Verifiable Visual RAG</h1>
            <p className="text-[11px] leading-tight text-faint">
              Region-level evidence, independently verified
            </p>
          </div>

          <form
            className="flex flex-1 items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              void run(question);
            }}
          >
            <div className="relative flex-1">
              <SearchIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-faint" />
              <label htmlFor="question" className="sr-only">
                Question
              </label>
              <input
                id="question"
                className="h-10 w-full rounded-full border border-border bg-surface pl-9 pr-4 text-sm outline-none transition-colors duration-150 placeholder:text-faint focus:border-accent"
                placeholder="Ask a question about an indexed document"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                disabled={pending}
                autoComplete="off"
              />
            </div>
            <button
              type="submit"
              disabled={pending || !question.trim()}
              className="h-10 shrink-0 cursor-pointer rounded-full bg-accent px-5 text-sm font-medium text-white transition-opacity duration-150 hover:opacity-90 disabled:cursor-default disabled:opacity-40"
            >
              {pending ? "Working" : "Ask"}
            </button>
          </form>
        </div>
        {pending && (
          <div className="h-0.5 w-full overflow-hidden bg-transparent">
            <div className="vv-sweep h-full w-1/4 bg-accent" />
          </div>
        )}
      </header>

      <main className="mx-auto w-full max-w-[1600px] flex-1 px-5 py-6 lg:px-8">
        {error && (
          <div
            role="alert"
            className="mb-5 flex gap-2.5 rounded-xl border border-danger/40 bg-danger/8 px-4 py-3 text-sm text-danger"
          >
            <AlertIcon className="mt-0.5 h-4 w-4 shrink-0" />
            <span className="min-w-0 break-words">{error}</span>
          </div>
        )}

        {!asked ? (
          <div className="mx-auto max-w-xl py-24 text-center">
            <h2 className="text-2xl font-semibold tracking-tight">
              Ask, and see exactly where the answer came from.
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              Every sentence of the answer is checked by a second model and pinned to a
              region of the page it came from. Claims that fail the check are withheld
              rather than shown.
            </p>
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)] lg:gap-8">
            <div className="min-w-0">
              <p aria-live="polite" className="sr-only">
                {status ?? (done ? "Answer complete" : "")}
              </p>

              {retrieved?.warning && (
                <div className="mb-4 flex gap-2.5 rounded-xl border border-warn/40 bg-warn-soft px-4 py-3 text-sm text-warn">
                  <AlertIcon className="mt-0.5 h-4 w-4 shrink-0" />
                  <span className="min-w-0">{retrieved.warning}</span>
                </div>
              )}

              {abstained && (
                <div className="mb-4 rounded-2xl border-2 border-warn/50 bg-warn-soft px-5 py-4">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-warn">
                    No answer given
                  </p>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted">
                    The verifier rejected every claim the reader produced, so the system is
                    declining to answer rather than showing something it cannot support.
                    {/* Without this, an unembedded page reads as a judgement about
                        the evidence when the real cause is that nobody ran
                        `vvrag embed`. */}
                    {retrieved?.warning &&
                      " Note the warning above: this page has no embeddings, so that" +
                        " rejection is very likely a missing index rather than weak evidence."}
                  </p>
                </div>
              )}

              {status && shown.length === 0 && (
                <div className="rounded-2xl border border-border bg-surface p-5">
                  <p className="text-sm text-muted">{status}</p>
                  <div className="mt-4 space-y-2.5" aria-hidden>
                    <div className="h-3 w-full rounded bg-sunken" />
                    <div className="h-3 w-[88%] rounded bg-sunken" />
                    <div className="h-3 w-[62%] rounded bg-sunken" />
                  </div>
                </div>
              )}

              {shown.length > 0 && (
                <AnswerPanel
                  shown={shown}
                  done={done}
                  hovered={hovered}
                  onHover={setHovered}
                  onSelect={revealClaim}
                />
              )}

              {status && shown.length > 0 && (
                <p className="mt-3 px-1 text-xs text-faint">{status}</p>
              )}

              <EvidenceVault
                shown={shown}
                withheld={withheld}
                imageUrl={imageUrl}
                expanded={expanded}
                hovered={hovered}
                onToggle={revealClaim}
                onHover={setHovered}
                onZoom={(claim, region) => setZoom({ claim, region })}
                pageAspect={pageAspect}
              />

              {done && !abstained && shown.length === 0 && (
                <p className="mt-3 text-sm text-muted">No claims to show.</p>
              )}
            </div>

            {/* Sticky on desktop so the page stays in view while the evidence
                vault scrolls beside it. The two are meant to be read against
                each other, and a viewer that scrolls away breaks that. */}
            <div className="min-w-0 lg:sticky lg:top-[5.5rem] lg:self-start">
              {retrieved && imageUrl ? (
                <>
                  <PageViewer
                    retrieved={retrieved}
                    imageUrl={imageUrl}
                    score={rankScore}
                    shown={shown}
                    hovered={hovered}
                    onHover={setHovered}
                    onZoom={(claim, region) => setZoom({ claim, region })}
                    onMeasure={setPageAspect}
                  />
                  <PageRail
                    candidates={alternates}
                    activeDoc={retrieved.doc_sha}
                    activePage={retrieved.page}
                    isRead={(sha, page) => results[keyOf(sha, page)] !== undefined}
                    pending={pending}
                    onSelect={(cand) =>
                      void run(asked, { doc: cand.doc_sha, page: cand.page })
                    }
                  />
                </>
              ) : (
                <div className="flex h-72 items-center justify-center rounded-2xl border border-dashed border-border text-sm text-faint">
                  The retrieved page appears here.
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {zoom && retrieved && imageUrl && (
        <ZoomOverlay
          imageUrl={imageUrl}
          claim={zoom.claim}
          region={zoom.region}
          docName={retrieved.doc_name}
          page={retrieved.page}
          pageAspect={pageAspect}
          onClose={() => setZoom(null)}
        />
      )}
    </>
  );
}
