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

\paragraph{Investigation of reverse pass deduplication}
It is conceivable that with sufficient massaging of inlining pass order,
GHC can spot duplicated calls to forward passes (see Section~\ref{recomp}) in
lens setters. This is a promising avenue that should fix a key limitation
of the existing framework.

\paragraph{Discrete and probabilistic CRDCs.}
The Cruttwell et al.\ framework is defined for any CRDC, not just
smooth Euclidean spaces.  Instantiating the Haskell library for a
discrete CRDC (e.g., a category of Boolean circuits) or a probabilistic
one would generalise gradient-based learning beyond the real-valued
setting and test the abstraction boundaries of the current design.

\section{Limitations}
\label{sec:limitations}

Several limitations of the current design are worth stating directly.

\paragraph{Forward-pass recomputation in the backward pass.}
\label{recomp}
The Van Laarhoven lens representation encodes the forward pass in the
getter and the backward pass in the setter as a single higher-rank
function.  This means the setter receives the original input and the
incoming gradient, but not the intermediate activations the getter
computed along the way.  For simple layers such as |linear| this is
costless, but |selfAttention| must recompute all six intermediate
tensors ($Q$, $K$, $V$, the score matrix, softmax weights, and attended
output) by calling |runFwd| a second time, performing two forward passes
per training step.  The root cause is structural: the lens
representation provides no channel for the getter to pass cached state
to the setter.  Replicating the autograd remedy (an explicit computation
graph whose nodes retain their outputs) would require encoding the
activation cache in the parameter type, significantly complicating the
type signatures.

\paragraph{Global optimiser state and step-dependent updates.}
The |repara| mechanism carries per-layer optimiser state on the
vertical parameter wire, which handles momentum buffers and accumulated
gradient statistics cleanly.  It does not, however, accommodate state
that is global across layers or that evolves with a step counter
independent of any individual layer.  Adam's bias-correction terms
$\hat{m} = m/(1-\beta_1^t)$ and $\hat{v} = v/(1-\beta_2^t)$ require
the current step $t$, which the current implementation omits
(Section~\ref{sec:adaptive}): including $t$ would require adding a step
counter to every layer's optimiser state or breaking the uniform
|repara| interface.  The same difficulty applies to learning-rate
schedules, gradient clipping by global norm, and weight decay as a
separate regularisation term.

\paragraph{Stateful and stochastic layers.}
Several standard layer types do not fit the deterministic, stateless
|ParaLens'| model.  Dropout requires sampling a random mask during the
forward pass and retaining it for the backward pass, which demands an
|IO| or |State| monad absent from the current type.  Batch
normalisation maintains running mean and variance statistics that are
updated during training but frozen at inference time, requiring a
training-mode flag that is not encoded in |ParaLens'|.  Layer
normalisation is deterministic and fits cleanly; the others do not.
Any model that relies on dropout or batch normalisation therefore cannot
be expressed in the current framework without a more substantial
extension to the core type.

\paragraph{Type-checker scaling.}
The type-level depth parameter uses Peano naturals and typeclass
induction.  At $n = 3$, |StackedN 3 m| reduces to |(m, (m, m))| at
compile time, which GHC can fully specialise and unbox.  GHC's
constraint solver is known to exhibit quadratic or worse behaviour on
deeply nested tuple types and long typeclass resolution chains in
general.  In practice, compilation times at $n = 50$ remained
acceptable, and Core size grows linearly with depth
(Section~\ref{sec:zero-overhead}), suggesting that the typeclass
resolution cost is also approximately linear for this particular
induction structure.  For substantially deeper networks---a 50-layer
ResNet with multiple sub-layers per block, for instance---compile-time
scaling is plausible but untested beyond the depths evaluated here.

\paragraph{Inliner budget and the zero-overhead guarantee.}
The zero-overhead property depends on GHC's simplifier fully inlining
and specialising the |forall f. Functor f =>| quantifier at each call
site and eliminating all the tuple isomorphisms (|swapFst|, |rotate|,
|rightLens|) that each |(.#.)| introduces.  GHC's simplifier operates
within a finite tick budget \texttt{-fsimpl-tick-factor};
each |(.#.)| composition adds several reduction steps, so a sufficiently
deep network could in principle exhaust this budget.  In practice,
the property has been verified at $n = 1$, $n = 20$ (1{,}800 lines of
Core), and $n = 50$ (4{,}900 lines of Core), with Core size growing
linearly and the lens abstraction absent from the output at every depth
tested (Section~\ref{sec:zero-overhead}). This suggests the guarantee
holds for any practically realisable network; should a sufficiently deep
model ever approach the default limit, raising
\texttt{-fsimpl-tick-factor} explicitly is a straightforward
mitigation.

\paragraph{Type error ergonomics.}
Even when the type system catches a mistake, the resulting error message
may be difficult to interpret.  Using local aliases to simplify
the types, passing a |Tensor [1,10]| where |Tensor [1,4]| is expected:

\begin{spec}
type Tensor s  = T.Tensor DV DT s
type MMP o i   = L.MMP DV DT o i

foo :: ParaLens' (MMP 6 4) (Tensor [b, 4]) (Tensor [b, 6])
foo = matMulLens

x :: Tensor [1, 6]
x = view foo (p :: MMP 6 4, oops :: Tensor [1, 10])
\end{spec}

\noindent nonetheless produces:

\begin{verbatim}
* Couldn't match type '10' with '4'
    arising from a functional dependency between:
      constraint 'mtl-2.3.1:Control.Monad.Reader.Class.MonadReader
                    (MMP 6 4, Tensor [1, 4]) ((->) (MMP 6 4, Tensor [1, 10]))'
        arising from a use of 'view'
      instance 'mtl-2.3.1:Control.Monad.Reader.Class.MonadReader
                  r ((->) r)'
        at <no location info>
* In the expression:
    view foo (p :: MMP 6 4, oops :: Tensor [1, 10])
  In an equation for 'x':
      x = view foo (p :: MMP 6 4, oops :: Tensor [1, 10])
\end{verbatim}

\noindent The core diagnosis (|Couldn't match type '10' with '4'|) is
correct, but it is buried inside a functional-dependency trace through
|mtl|'s |MonadReader|, an artefact of the Van Laarhoven encoding that a
user should not need to understand.  Errors from failed type family
unification in |ConvSideCheck| or |StackedN| are more verbose still.
Using the framework's own runner (|runFullModel|) yields cleaner
messages, but the ergonomics cost remains and affects accessibility for
practitioners unfamiliar with GHC's type system.

\paragraph{Benchmark scope.}
All performance results are from a CPU-only machine with toy models.
The $5.3\times$ speedup over dynamic autograd is measured on a
two-layer Iris MLP with 1{,}527 parameters and batch size 8.  On
GPU workloads with large models the cost of PyTorch's autograd tape
construction shrinks relative to the tensor arithmetic itself, so the
advantage is likely smaller or absent at scale.  The claim that the
typed lens implementation will match HMatrix for larger matrices
(Section~\ref{sec:perf}) is plausible given the per-call FFI
overhead analysis, but is asserted rather than measured.

\section{Broader Impact}

One key benefit of this work is the promotion of ML configuration errors (e.g. shape
mismatches, invalid device/dtype combinations, architectural depth
mistakes) from silent runtime failures to compile-time rejections.
This manifested concretely during development: porting an untyped
prototype to the typed implementation exposed asymmetric index errors in
several backward passes that had gone undetected because tests used equal
input and output dimensions; the type system rejected them immediately.

The guarantees are nonetheless narrow: they address gradient mechanics
and tensor bookkeeping, not deployment behaviour.  A well-typed network
trained on biased data remains biased.  There is also an access barrier:
the safety properties described here require Haskell fluency, gating the
benefits behind a significant learning curve and reinforcing existing
stratification in who can audit ML infrastructure.  A productive
long-term direction is to export these ideas to mainstream frameworks via
type-level extensions to Python or automatic generation of typed
interfaces, making compile-time shape safety accessible without a change
of language.