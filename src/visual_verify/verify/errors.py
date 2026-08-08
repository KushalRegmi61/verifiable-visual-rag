class VerifierError(RuntimeError):
    """A contract violation or transport failure in the read-verify pipeline.

    Own module on purpose. claims, rubric, and core all raise this, and
    putting it in core or __init__ makes one of them import a sibling that
    imports it back, a cycle that only works by definition order.
    """
