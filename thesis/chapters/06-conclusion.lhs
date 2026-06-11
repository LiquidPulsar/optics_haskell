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
gradient-based machine learning.  The central abstraction, that every
component of a training pipeline is a parametric lens carrying a forward
pass and a gradient pass as a single composable unit, is realised here
with full static verification.  Tensor shapes, mini-batch sizes, compute
devices, and element dtypes are all encoded as type-level parameters, so
a misconfigured architecture is rejected by the type checker rather than
failing at runtime or producing a silent numerical error.

The library extends the original Python proof-of-concept in eight
concrete directions.  First, all implementations are written in batched
form from the ground up: every layer, loss function, and optimiser
operates over a leading batch dimension $b$ that is a type-level
\texttt{Nat}, whereas the original Cruttwell et al.\ implementation
processes a single example at a time.  Second, all tensor dimensions are
static: GHC's \texttt{DataKinds} extension encodes every dimension as a
type-level \texttt{Nat}, and type families compute output shapes for
convolutions and pooling layers automatically, making spatial mismatches
compile-time errors (Chapter~\ref{chap:layers}).  Third, device and
dtype polymorphism are built into every model signature: a single
|ParaLens'| definition is valid on CPU or CUDA and in 32-bit or 64-bit
floating point, with constraint synonyms such as |SaneDT| preventing
invalid device/dtype combinations from type-checking.  Fourth, the
framework is extended beyond the MLP scope of the original paper to
autoencoder architectures: encoder and decoder are each ordinary
|ParaLens'| pipelines that compose with |(.#.)| into a single
end-to-end model, trained on MNIST images with MSE reconstruction loss.
Fifth, scaled dot-product self-attention and its multi-head
generalisation are implemented as |ParaLens'| values with fully manually
derived backward passes (including the rank-one Jacobian correction for
the softmax non-linearity) extending the framework to the attention
mechanism that underlies modern transformer architectures
(Chapter~\ref{chap:layers}).  Sixth, residual skip connections are
realised as a single higher-order combinator |skipPara|, built entirely
from existing lens primitives with no new axioms; the identity gradient
path that ensures gradients reach early layers even when the learned
branch saturates emerges automatically from the combinator structure,
requiring no separate backward-pass derivation (Section~\ref{sec:skipconnections}).
Seventh, type-level Peano naturals and typeclass induction are used to
build networks of variable depth $n$ whose parameter type is a fully
concrete nested tuple at each $n$, preserving GHC's ability to
specialise and unbox the entire parameter structure and yielding zero
overhead relative to a hand-written network of the same depth.  Eighth,
the zero-overhead property is verified directly by GHC Core inspection:
compiling the lens-based forward pass and an equivalent hand-written
forward pass with \texttt{-O2 -ddump-simpl} produces structurally
identical worker functions, confirming that the entire |ParaLens|
abstraction: composition, reparametrisation, the van Laarhoven
|forall|; is absent from the optimised output
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

\section{Broader Impact}

The primary benefit of this work is the promotion of a class of ML 
configuration errors (shape mismatches, invalid device/dtype combinations, 
architectural depth mistakes) from silent runtime failures to compile-time 
rejections. In production or safety-critical deployments, such errors 
currently surface only when code runs, at which point significant 
computation may already have been wasted or, in embedded settings, harm 
caused. Making them structurally impossible narrows the gap between what 
a model is intended to compute and what it actually computes. That said, 
the guarantees offered are narrow in scope: they address the mechanics 
of gradient flow and tensor bookkeeping, not the behaviour of the resulting 
model in deployment. A well-typed network trained on biased or 
unrepresentative data remains biased; a shape-correct architecture can 
produce confidently wrong predictions on out-of-distribution inputs. 
The framework says nothing about data quality, label fairness, or 
distributional shift — the issues that dominate the ethical literature 
on ML systems. There is also an access consideration: the safety properties 
described here are available only to practitioners fluent in Haskell, a 
language with a substantially smaller and less diverse community than Python. 
This is not ethically neutral; it means the benefits of static verification 
are currently gated behind a significant learning barrier, reinforcing 
existing stratification in who can build and audit ML infrastructure. A 
productive long-term direction would be to export these ideas to mainstream 
frameworks — whether through type-level extensions to Python's type system 
or by generating typed interfaces automatically from existing model 
definitions — making compile-time shape safety accessible without requiring 
a change of language.