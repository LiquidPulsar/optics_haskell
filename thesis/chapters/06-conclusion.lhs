%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%format ->    = "\to"
%format =>    = "\Rightarrow"
%format forall = "\forall"
%format .#.   = "\mathbin{\bullet}"
%format ***   = "\mathbin{\times}"
%format <$>   = "\mathbin{\langle\$\rangle}"
%format @     = "\mathbin{@}"
%% ─────────────────────────────────────────────────────────────────────────────

%if False
\begin{code}
module Conclusion where
\end{code}
%endif

\chapter{Conclusion}
\label{chap:conclusion}

\section{Summary}

This thesis has instantiated the categorical framework of Cruttwell
et al.~\citep{catlearning} as a working Haskell library for
gradient-based machine learning.  The central abstraction---that every
component of a training pipeline is a parametric lens carrying a forward
pass and a gradient pass as a single composable unit---is realised here
with full static verification.  Tensor shapes, mini-batch sizes, compute
devices, and element dtypes are all encoded as type-level parameters, so
a misconfigured architecture is rejected by the type checker rather than
failing at runtime or producing a silent numerical error.

The library extends the original Python proof-of-concept in five concrete
directions.  First, all implementations are written in batched form from
the ground up: every layer, loss function, and optimiser operates over a
leading batch dimension $b$ that is a type-level \texttt{Nat}, whereas
the original Cruttwell et al.\ implementation processes a single example
at a time.  Second, all tensor dimensions are static: GHC's
\texttt{DataKinds} extension encodes every dimension as a type-level
\texttt{Nat}, and type families compute output shapes for convolutions
and pooling layers automatically, making spatial mismatches compile-time
errors (Chapter~\ref{chap:layers}).  Third, device and dtype polymorphism
are built into every model signature: a single |ParaLens'| definition is
valid on CPU or CUDA and in 32-bit or 64-bit floating point, with
constraint synonyms such as |SaneDT| preventing invalid device/dtype
combinations from type-checking.  Fourth, the framework is extended
beyond the MLP scope of the original paper to autoencoder architectures:
encoder and decoder are each ordinary |ParaLens'| pipelines that compose
with |(.#.)| into a single end-to-end model, trained on MNIST images
with MSE reconstruction loss.  Fifth, scaled dot-product self-attention
and its multi-head generalisation are implemented as |ParaLens'| values
with fully manually derived backward passes---including the rank-one
Jacobian correction for the softmax non-linearity---extending the
framework to the attention mechanism that underlies modern transformer
architectures (Chapter~\ref{chap:layers}).

The zero-overhead property of the abstraction is verified directly by
GHC Core inspection.  Compiling the Iris lens-based model alongside an
equivalent hand-rolled forward pass with \texttt{-O2 -ddump-simpl}
produces structurally identical worker functions: the entire |ParaLens|
machinery---composition, reparametrisation, the van Laarhoven
|forall|---is absent from the optimised output
(Section~\ref{sec:zero-overhead}).

\section{Future Work}

Several directions remain open for future investigation.

\paragraph{Recurrent architectures.}
Recurrent networks maintain hidden state across time steps, which does
not fit the stateless composition pattern of |(.#.)| directly.  One
avenue is to treat the hidden state as an additional parameter, making
the recurrence an instance of |repara| on the state wire.  Whether
this recovers standard BPTT or requires a categorical extension remains
an open question.

\paragraph{Formal verification of lens laws.}
The lens laws (get-put, put-get, put-put) are satisfied structurally
by construction in this framework, but are not machine-verified.
Property-based testing with QuickCheck could confirm the laws hold for
all implemented lenses under arbitrary inputs, providing stronger
assurance than the informal argument from construction.

\paragraph{Discrete and probabilistic CRDCs.}
The Cruttwell et al.\ framework is defined for any CRDC, not just
smooth Euclidean spaces.  Instantiating the Haskell library for a
discrete CRDC (e.g., a category of Boolean circuits) or a probabilistic
one would generalise gradient-based learning beyond the real-valued
setting and test the abstraction boundaries of the current design.
