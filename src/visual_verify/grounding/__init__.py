"""Region-level grounding: pillar 2 of the project.

The heatmap RANKS candidate boxes that already exist in the document text
layer. It never draws one. Every bbox this package returns is, exactly, the
bbox of a BoxRecord that came out of derive.py.

Pure numpy over arguments: page vectors, query vectors, and the patch grid are
passed in, never fetched. That keeps the package inside the core's four
dependencies and makes the whole ranking path testable with hand-built arrays,
with no Qdrant, no GPU, and no 21.4 s/page model load.

Task 7 adds the public re-exports here. Nothing else belongs in this file:
defining a shared symbol here and importing it back from a submodule creates a
cycle that only works by definition order, which is a trap for whoever edits
it next.
"""
