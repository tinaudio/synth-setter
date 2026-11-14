"""Shared query parameters for approximate-nearest-neighbour assertions."""

# Candidate multiplier that re-ranks IVF_PQ hits against the exact vectors,
# which lossy PQ codes alone cannot order reliably.
ANN_SELF_QUERY_REFINE_FACTOR = 10
