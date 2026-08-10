"use client";

import { useLayoutEffect, useRef, type KeyboardEvent } from "react";
import { SearchIcon, SendIcon } from "./icons";

// About seven lines. Past that the composer stops growing and scrolls, because
// a header that keeps expanding eventually covers the answer it sits above, and
// this one is sticky.
const MAX_HEIGHT = 168;

type Props = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  pending: boolean;
};

/**
 * The question box, growing with what is typed.
 *
 * A single-line input was wrong for the input this system actually takes.
 * Questions worth asking a document are sentences, and a one-line field
 * scrolls the beginning of the question out of sight exactly when the user
 * wants to reread it before committing to a run that costs a GPU embed, a
 * reader call and a verifier call per claim.
 *
 * The border and background live on the WRAPPER, not the textarea. Auto-growth
 * measures scrollHeight, which under border-box includes padding but excludes
 * borders, so a bordered textarea grows two pixels short on every keystroke and
 * the last line creeps out of view.
 */
export function AskComposer({ value, onChange, onSubmit, pending }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    const fit = () => {
      // Reset first. scrollHeight can never report less than the current
      // height, so without this the box only ever grows and deleting a line
      // leaves a gap.
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
      el.style.overflowY = el.scrollHeight > MAX_HEIGHT ? "auto" : "hidden";
    };

    fit();
    // The height is a function of the WIDTH as well as the text, and the width
    // changes without the text changing: the window resizes, and the button
    // itself switches between a 44px square and a wider labelled one at 640px.
    // Without this the inline height survives the resize and a one-line
    // question sits in a two-line box. Found by resizing the window, not by a
    // test. A ResizeObserver on the field itself would be the obvious tool and
    // is the wrong one, because this handler resizes that same element.
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [value]);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key !== "Enter") return;
    // An IME candidate window uses Enter to accept a suggestion. Submitting
    // there would send a half-composed question and discard the rest.
    if (e.nativeEvent.isComposing) return;
    // On a touch keyboard there is no Shift+Enter, so Enter has to mean a new
    // line and the button is the only way to submit. Checked at press time
    // rather than at render, because reading matchMedia during render would
    // disagree with the server-rendered markup.
    const touch = window.matchMedia("(pointer: coarse)").matches;
    if (touch || e.shiftKey) return;
    e.preventDefault();
    onSubmit();
  }

  return (
    <form
      className="flex-1"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor="question" className="sr-only">
        Question
      </label>
      {/* items-start, so nothing here moves as the field grows. Pinning the
          button to the LAST line meant it slid down the screen while the user
          was still typing the question it submits, which reads as the layout
          coming apart rather than as the field doing its job. The icon and the
          button now frame the first line and stay there; only the text moves.
          One row at every width, so growth never reflows the row either. */}
      <div className="flex items-start gap-2 rounded-[22px] border border-border bg-surface px-3 py-1.5 transition-colors duration-150 focus-within:border-accent">
        {/* mt-2.5 puts it on the optical centre of the first line rather than
            the top of the padding box. */}
        <SearchIcon className="mt-2.5 h-4 w-4 shrink-0 text-faint" />
        <textarea
          id="question"
          ref={ref}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={pending}
          placeholder="Ask a question about an indexed document"
          aria-describedby="question-hint"
          autoComplete="off"
          // resize-none because the drag handle would fight the auto-growth,
          // and leading-6 so the measured height lands on whole lines.
          className="max-h-[168px] min-h-[28px] flex-1 resize-none bg-transparent py-1 text-sm leading-6 outline-none placeholder:text-faint disabled:opacity-60"
        />
        {/* Icon-only below 640px. Spelling out "Ask" there costs about 50px
            of a 375px screen and left the field too narrow to reread a
            sentence in, which is what made the button wrap onto its own row
            before, which is what made it move. 44px square on touch, per the
            minimum target size. */}
        <button
          type="submit"
          disabled={pending || !value.trim()}
          aria-label={pending ? "Working" : "Ask"}
          className="inline-flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-full bg-accent text-sm font-medium text-white transition-opacity duration-150 hover:opacity-90 disabled:cursor-default disabled:opacity-40 sm:h-9 sm:w-auto sm:px-5"
        >
          <SendIcon className="h-4 w-4 sm:hidden" />
          <span className="hidden sm:inline">{pending ? "Working" : "Ask"}</span>
        </button>
      </div>
      <p id="question-hint" className="sr-only">
        Press Enter to ask, or Shift and Enter for a new line.
      </p>
    </form>
  );
}
