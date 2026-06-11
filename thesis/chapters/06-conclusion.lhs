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

\section{Limitations}
\label{sec:limitations}

Several limitations of the current design are worth stating directly.

\paragraph{Forward-pass recomputation in the backward pass.}
The Van Laarhoven lens representation encodes the forward pass in the
getter and the backward pass in the setter as a single higher-rank
function.  This means the setter receives the original input and the
incoming gradient, but not the intermediate activations that the getter
computed along the way.  For simple layers such as |linear|, this is
costless: the backward pass for $xW^\top$ needs only $x$ and
$\partial L/\partial y$, both already in scope.  For
|selfAttention|, however, the setter must recompute all six
intermediate tensors ($Q$, $K$, $V$, the score matrix, the softmax
weights, and the attended output) by calling |runFwd| again:

\begin{spec}
rev (p @ (wq, wk, wv, wo), x) dOut = ...
  where
    (q, k, v, weights, attn, _) = runFwd p x
\end{spec}

\noindent This means attention performs two forward passes per training
step.  The root cause is
structural: the lens representation does not provide a natural channel
for the getter to pass cached state to the setter.  The standard remedy
in autograd frameworks is an explicit computation graph whose nodes
retain their intermediate outputs; replicating that within the lens
abstraction would require encoding the activation cache in the parameter
type or in a separate residual state wire, significantly complicating
the type signatures.  All layers in the current implementation that
require caching share this limitation.

\paragraph{Global optimiser state and step-dependent updates.}
The |repara| mechanism carries per-layer optimiser state on the
vertical parameter wire, which handles momentum buffers and accumulated
gradient statistics cleanly.  It does not, however, accommodate state
that is global across layers or that evolves with a step counter
independent of any individual layer.  Adam's bias-correction terms

\[
  \hat m = \frac{m}{1 - \beta_1^t}, \qquad \hat v = \frac{v}{1 - \beta_2^t}
\]

\noindent require the current step $t$, which the current implementation
omits (Section~\ref{sec:optimisers}).  Including $t$ would require
adding a step counter to every layer's optimiser state, polluting every
type signature, or passing it as an external argument to the update
rule, breaking the uniform |repara| interface.  The same difficulty
applies to learning-rate schedules, gradient clipping by global norm,
and weight decay treated as a regularisation term separate from the main
gradient: all require either global coordination or information spanning
multiple layers, neither of which the per-layer compositionality of
|repara| naturally supports.

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
deeply nested tuple types and long typeclass resolution chains, however.
The thesis evaluates only small models (at most three residual blocks).
For realistically deep networks---a 50-layer ResNet, for
instance---compilation time and type-checker memory usage are unknown
and potentially prohibitive.  The zero-overhead property at runtime may
therefore come at the cost of overhead at compile time that does not
scale gracefully.

\paragraph{Inliner budget and the zero-overhead guarantee.}
The zero-overhead property depends on GHC's simplifier fully inlining
and specialising the |forall f. Functor f =>| quantifier at each call
site, and on eliminating all the tuple isomorphisms (|swapFst|,
|rotate|, |rightLens|) that each |(•)| introduces.  GHC's simplifier
operates within a finite tick budget controlled by
\texttt{-fsimpl-tick-factor} (default 100); when the budget is
exhausted it emits a \emph{Simplifier ticks exhausted} warning and
stops, leaving residual |fmap| calls and tuple constructors in the
output.  Because each |(•)| composition adds several reduction steps,
a sufficiently deep network could exhaust this budget before the
|forall f| is specialised away, at which point the abstraction survives
into the compiled binary and the zero-overhead claim no longer holds.
A secondary mechanism reinforces this risk: GHC's inliner avoids
inlining expressions above a size threshold, so if the composed lens
term grows large enough before specialisation, |repara| and |stackN|
calls may be retained even before the tick limit is reached.  The
thesis verifies zero overhead only for the two-layer Iris model
($n = 1$); the property is asserted but not tested for deeper
architectures.  For deep networks one mitigation would be to raise
\texttt{-fsimpl-tick-factor} explicitly or to introduce strategic
|SPECIALIZE| pragmas at fixed depths, but neither is currently
explored.

\paragraph{Type error ergonomics.}
The thesis demonstrates the type system catching shape mismatches, but
shows only the success case.  In practice, GHC error messages for
failed type family unification involving |ConvSideCheck|, |StackedN|,
or |ToPeano| can be verbose and difficult to interpret: a wrong kernel
size produces a constraint-solver failure that does not directly name
the mismatched dimension.  This is an ergonomics cost that partially
offsets the safety benefit and affects how accessible the framework is
to practitioners not already familiar with GHC's type system.

\paragraph{Benchmark scope.}
All performance results are from a CPU-only machine with toy models.
The $5.3\times$ speedup over dynamic autograd is measured on a
two-layer Iris MLP with 1{,}527 parameters and batch size 8.  On
GPU workloads with large models the cost of PyTorch's autograd tape
construction shrinks relative to the tensor arithmetic itself, so the
advantage is likely smaller or absent at scale.  The claim that the
typed lens implementation will match HMatrix for larger matrices
(Section~\ref{sec:performance}) is plausible given the per-call FFI
overhead analysis, but is asserted rather than measured.

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
distributional shift: the issues that dominate the ethical literature 
on ML systems. There is also an access consideration: the safety properties 
described here are available only to practitioners fluent in Haskell, a 
language with a substantially smaller and less diverse community than Python. 
This is not ethically neutral; it means the benefits of static verification 
are currently gated behind a significant learning barrier, reinforcing 
existing stratification in who can build and audit ML infrastructure. A 
productive long-term direction would be to export these ideas to mainstream 
frameworks, whether through type-level extensions to Python's type system 
or by generating typed interfaces automatically from existing model 
definitions, thus making compile-time shape safety accessible without requiring 
a change of language.