"use client";

import type { ClaimEvent, DoneEvent } from "@/lib/api";
import { colourFor, groupIntoParagraphs } from "@/lib/claims";

type Props = {
  shown: ClaimEvent[];
  done: DoneEvent | null;
  // Its own prop rather than `done.withheld`, which this panel already
  // receives. `done` only arrives when the stream finishes, so a count read
  // from it would sit at zero for the whole verification pass and then jump at
  // the end, which reads as the system having changed its mind. The live count
  // comes from the claims already in hand.
  withheldCount: number;
  hovered: number | null;
  onHover: (index: number | null) => void;
  onSelect: (index: number) => void;
};

/**
 * The answer, in its own section, as prose.
 *
 * It is the verified claims joined and nothing else. Deliberately NOT a second
 * model call that summarises them: a synthesised sentence would be unverified
 * text presented as the answer, which is the exact failure the verifier and the
 * abstention gate exist to prevent. reader.py states the rule, "there is no
 * separate prose answer, the displayed answer is the claims joined, so nothing
 * can drift between what is shown and what is verified"; this is where that
 * becomes visible.
 *
 * Every sentence is its own control: hovering lights its region on the page,
 * clicking opens its evidence. That is what makes the join honest rather than
 * merely compact, because the seam between two claims stays visible and
 * traversable instead of being smoothed into one paragraph of unattributed
 * text.
 */
export function AnswerPanel({
  shown,
  done,
  withheldCount,
  hovered,
  onHover,
  onSelect,
}: Props) {
  return (
    <section
      aria-labelledby="answer-heading"
      className="rounded-2xl border border-border bg-surface p-5 shadow-sm sm:p-6"
    >
      <div className="flex items-baseline justify-between gap-4">
        <h2
          id="answer-heading"
          className="text-[11px] font-semibold uppercase tracking-[0.14em] text-faint"
        >
          Answer
        </h2>
        {done && (
          <p className="text-[11px] text-faint tnum">
            {done.shown} verified
            {done.withheld > 0 && `, ${done.withheld} withheld`}
          </p>
        )}
      </div>

      {groupIntoParagraphs(shown).map((paragraph, i) => (
        <p
          key={paragraph[0].index}
          className={`text-[17px] leading-[1.65] tracking-[-0.006em] sm:text-lg ${
            i === 0 ? "mt-3" : "mt-4"
          }`}
        >
          {paragraph.map((c) => {
            const colour = colourFor(c.index);
            const dimmed = hovered !== null && hovered !== c.index;
            return (
              <span
                key={c.index}
                // A span, not a button: a button inside a paragraph cannot wrap
                // across lines without breaking the text flow, and this text must
                // read as a paragraph first. Keyboard users reach the same claim
                // through the evidence vault, where it is a real disclosure
                // control.
                onMouseEnter={() => onHover(c.index)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(c.index)}
                style={{
                  textDecorationColor: colour,
                  opacity: dimmed ? 0.38 : 1,
                }}
                className="cursor-pointer underline decoration-2 underline-offset-[5px] transition-opacity duration-150"
              >
                {c.text}
                {/* The ordinal, not decoration. Without it the only thing tying a
                    sentence to its evidence is the underline hue, which fails for
                    anyone who cannot separate the colours and is close to
                    invisible on the dark card anyway. It reads as a citation
                    marker, which is what it is. */}
                <sup
                  className="ml-0.5 text-[10px] font-semibold tnum"
                  style={{ color: colour }}
                >
                  {c.index + 1}
                </sup>{" "}
              </span>
            );
          })}
        </p>
      ))}

      {/* Says only what this system did, never what the page contains. "I could
          not confirm it" is introspection and is true by construction whenever
          it is shown; "the page does not say so" would be a claim about the
          world with no region behind it. The header keeps its `N verified, M
          withheld` chip, which is the technical reading of the same fact; this
          is the conversational one, and both are wanted. */}
      {withheldCount > 0 && (
        <p className="mt-4 text-sm italic text-muted">
          I left out {withheldCount} statement{withheldCount === 1 ? "" : "s"} I could not
          confirm against this page.
        </p>
      )}

      <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-faint">
        Each sentence above was checked by a second model and carries its own
        evidence region. Hover one to find it on the page, or click it to open the
        evidence.
      </p>
    </section>
  );
}
