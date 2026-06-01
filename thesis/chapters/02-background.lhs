%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%format ->    = "\to"
%format =>    = "\Rightarrow"
%format forall = "\forall"
%format .#.   = "\mathbin{\bullet}"
%format <$>   = "\mathbin{\langle\$\rangle}"
%format @     = "\mathbin{@}"
%% ─────────────────────────────────────────────────────────────────────────────

%if False
\begin{code}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ConstraintKinds #-}
module Background where
\end{code}
%endif

\chapter{Background}
\label{chap:background}

This chapter introduces the three bodies of knowledge that underpin the
thesis: functional programming in Haskell, including the advanced type
system features that enable static shape checking; the theory of optics
that supplies the mathematical objects for representing neural network
layers; and the fundamentals of gradient-based learning together with
the categorical formalism used to reason about it.

\section{Functional Programming in Haskell}
\label{sec:haskell}

Haskell is a purely functional, statically typed language with lazy
evaluation.  \emph{Pure} functions have no side effects: a function
|f :: a -> b| maps every value of type |a| to a value of type |b|
without reading or writing global state.  Purity makes programs
compositional by default---the behaviour of a composition |f . g| is
fully determined by the behaviours of |f| and |g| individually---which
is the property the framework exploits to build large networks from
small verified components.

\subsection*{Type Classes}

The primary abstraction mechanism is the \emph{type class}.  A class
declares an interface; instances provide concrete implementations:

\begin{code}
class Functor f where
    fmap :: (a -> b) -> f a -> f b
\end{code}

\noindent Type class resolution is purely compile-time: there is no
dynamic dispatch.  This means that when |fmap| is specialised to a
concrete |f|, GHC can inline the specific implementation and optimise
it away.  This property is crucial in Section~\ref{sec:zero-overhead}
where the entire lens abstraction is shown to vanish in the compiled
output.

\subsection*{The Kind System and DataKinds}

Every type in Haskell has a \emph{kind}, which is the ``type of a
type.''  Ordinary types such as |Int| have kind |Type|.  A type
constructor such as |Maybe| has kind |Type -> Type|: it takes one type
argument before producing a concrete type.

The |{-# LANGUAGE DataKinds #-}| extension promotes value-level data
constructors to the type level.  The standard natural numbers, for
instance, are promoted to kind |Nat|, and lists of naturals to kind
|[Nat]|.  The typed tensor library Hasktorch exploits this directly:

\begin{code}
-- A rank-2 tensor on device dv, with element type dt,
-- holding a matrix of m rows and n columns:
example :: T.Tensor dv dt [m, n]
\end{code}

\noindent All three properties---device |dv|, element dtype |dt|, and
shape |[m, n]|---are resolved at compile time.  An attempt to multiply
a $[3,4]$ matrix by a $[5,6]$ matrix produces a type error before any
code runs.  No runtime shape assertions are needed.  Throughout this
thesis, type-level lists of |Nat| appear wherever tensor shapes are
tracked: kernel sizes |[kH, kW]|, convolution output dimensions, and
batch sizes |[b, d]| are all compile-time entities.

\subsection*{Type Families}

A \emph{type family} is a type-level function computed by the
compiler.  The |{-# LANGUAGE TypeFamilies #-}| extension enables
pattern-matching on types in closed equations:

\begin{code}
type family ToPeano (n :: Nat) :: PeanoNat where
    ToPeano 1  =  One
    ToPeano n  =  Succ (ToPeano (n - 1))
\end{code}

\noindent The compiler reduces |ToPeano 3| to |Succ (Succ One)| at
compile time, with no runtime cost.  Type families appear throughout
the framework for computing output shapes from layer parameters: the
output height of a convolution with kernel $k$, stride $s$, and
padding $p$ applied to an input of height $h$ is
$\lfloor (h - k + 2p)/s \rfloor + 1$, and this computation is
performed by a type family so that the output shape is available for
type-checking the next layer.

\subsection*{Higher-Rank Polymorphism}

Standard Haskell polymorphism places the |forall| at the outermost
level of a type: |id :: forall a. a -> a|.  The
|{-# LANGUAGE RankNTypes #-}| extension allows the quantifier to
appear inside a type:

\begin{code}
type Lens s t a b = forall f. Functor f => (a -> f b) -> s -> f t
\end{code}

\noindent Here |forall f| is scoped inside the synonym: any value of
type |Lens s t a b| must work for \emph{any} |Functor| the caller
chooses.  This is the van Laarhoven representation described in
Section~\ref{sec:optics-bg}.  The higher-rank quantifier is the reason
lenses compose with plain |(.)| --- the same function composes for any
functor --- and is also the mechanism by which the lens abstraction
collapses to direct function calls when the functor is fixed at the
call site (Section~\ref{sec:zero-overhead}).

\subsection*{Constraint Synonyms}

The |{-# LANGUAGE ConstraintKinds #-}| extension allows the kind
|Constraint| (the kind of type class requirements) to be treated as a
first-class kind.  This enables \emph{constraint synonyms}:

\begin{code}
type SaneDT dv dt  =
  (  T.KnownDType dt
  ,  T.StandardFloatingPointDTypeValidation dv dt
  ,  CanMMLens dv dt  )
\end{code}

\noindent |SaneDT dv dt| is a single alias for a conjunction of three
constraints.  Functions that require all three can write |SaneDT dv dt
=>| in their context rather than enumerating each constraint
separately.  Constraint synonyms keep type signatures concise, and the
type checker verifies the full set when a concrete |dv| and |dt| are
supplied.

\section{Optics}
\label{sec:optics-bg}

An \emph{optic} is a composable interface for accessing and modifying
a sub-component of a data structure.  The framework's full theory of
optics occupies Section~\ref{sec:lenses} of this chapter; this section
provides the intuition needed to follow the design.

\subsection*{The Concrete Lens}

The simplest optic is a \emph{lens}, which focuses on a single
sub-component.  Concretely, a lens is a pair of functions:

\begin{code}
data Lens s a = Lens
  { view :: s -> a
  , set  :: s -> a -> s }
\end{code}

\noindent |view| retrieves the focused value; |set| replaces it.  This
representation is straightforward but does not compose directly:
chaining two lenses requires manually threading the getter and setter
functions, producing boilerplate that grows with nesting depth.

\subsection*{The Van Laarhoven Representation}

Van Laarhoven~\citep{van2009lens} showed that both operations can be
unified into a single higher-rank function.  The full \emph{polymorphic
lens} type is:

\begin{code}
type Lens s t a b = forall f. Functor f => (a -> f b) -> s -> f t
\end{code}

\noindent where |s| is the source type, |t| the updated source, |a|
the focused value, and |b| its replacement.  The |forall f| is the
key: the choice of functor determines which operation is performed.
\begin{itemize}
  \item \textbf{Viewing:} with $f = \mathtt{Const}\;a$,
    $\mathtt{fmap} = \mathtt{const}$, the chain collapses to $s \to
    a$.
  \item \textbf{Setting:} with $f = \mathtt{Identity}$,
    $\mathtt{fmap} = \mathtt{id}$, the chain collapses to $(s, b) \to
    t$.
\end{itemize}
\noindent Because a van Laarhoven lens is simply a function of type
|(a -> f b) -> s -> f t|, two lenses |l1 :: Lens s t a b| and
|l2 :: Lens a b c d| compose by ordinary function composition:
\[
  \mathtt{l1\;.\;l2\;::\;Lens\;s\;t\;c\;d}.
\]
This composability is the central reason lenses are used throughout the
framework.  The |lens| library~\citep{ekmett2025lens} provides hundreds
of lenses for standard data structures that compose freely via |(.|).

\subsection*{Lens Laws}

A well-formed lens satisfies three equational laws:
\begin{enumerate}
  \item \textbf{Get-put:} |set s (view s) = s|.
  \item \textbf{Put-get:} |view (set s b) = b|.
  \item \textbf{Put-put:} |set (set s b) b' = set s b'|.
\end{enumerate}
\noindent These ensure the focused sub-component is genuinely
independent of the rest of the structure.  All lenses in this
framework are constructed from primitives that satisfy the laws
structurally.

\section{Neural Networks and Gradient-Based Learning}
\label{sec:nn-bg}

A \emph{neural network} is a parametric function $f_\theta : A \to B$
indexed by a weight vector $\theta \in P$.  The network is composed
from differentiable building blocks---affine maps, activation
functions, normalisation layers---each with their own local parameters.
Training adjusts $\theta$ to minimise a scalar loss
$\mathcal{L}(f_\theta(x), y)$ over a dataset of labelled pairs
$(x, y)$.

\subsection*{Gradient Descent}

When $P$ is a real vector space and $\mathcal{L}$ is differentiable,
the canonical update rule is gradient descent:
\[
  \theta \;\leftarrow\; \theta - \eta \cdot \nabla_\theta \mathcal{L},
  \qquad \eta > 0.
\]
Each step moves $\theta$ in the direction of steepest loss decrease.
Practical optimisers---momentum, Adam, AdaGrad---modify this rule to
improve convergence, but all share the same structure: a function from
current parameters and a gradient signal to updated parameters.  In
this framework that function is itself a lens, described in
Chapter~\ref{chap:optim}.

\subsection*{Backpropagation}

For a composed model $f_\theta = f^{(L)}_{\theta_L} \circ \cdots \circ
f^{(1)}_{\theta_1}$, computing $\nabla_\theta \mathcal{L}$ by the chain
rule is \emph{backpropagation}.  The forward pass evaluates $f_\theta(x)$
layer by layer, caching intermediate activations; the backward pass
propagates a gradient signal from the output back through each layer in
reverse, computing $\partial \mathcal{L}/\partial \theta_\ell$ at each
step.

In this framework both passes are encoded simultaneously in a single
lens: the \emph{getter} is the forward pass and the
\emph{setter--getter} pair is the backward pass.  The chain rule is
realised by lens composition via |(.)| and |(.#.)|.  No separate
backward-pass data structure is needed.

\subsection*{Mini-Batching}

In practice, the gradient is estimated on a \emph{mini-batch}: a small
subset of the dataset drawn at each step.  If each sample has shape
$[d]$, a batch of $b$ samples has shape $[b, d]$, and all $b$ forward
and backward passes are performed simultaneously as a single tensor
operation.  GPU hardware is optimised for this regular parallelism,
making batching essential for performance.

In this framework the batch size $b$ is a type-level |Nat|.  A model
typed to accept |T.Tensor dv dt [b, d]| cannot be accidentally applied
to a tensor of a different batch size: the mismatch is a compile-time
type error rather than a silent shape broadcast.

\section{Cartesian Reverse Differential Categories}
\label{sec:crdc-bg}

The theoretical foundation for backpropagation in this framework is the
notion of a \emph{Cartesian Reverse Differential Category}
(CRDC)~\citep{cruttwell2022}.  CRDCs provide an axiomatic account of
reverse-mode automatic differentiation that is independent of any
particular programming language or number system.

The key property used in this thesis is that every object $A$ in a
CRDC carries a \emph{natural addition} $+_A : A \times A \to A$,
arising from the abelian group structure that CRDCs impose.  In the
standard category of Euclidean spaces this is ordinary vector addition;
in a category of boolean circuits it would be a different, discrete
operation---the abstraction is independent of the specific category.

The training loop performs precisely this natural addition:
\[
  \theta_{\text{new}} \;=\; \theta_{\text{old}} + \partial\theta,
\]
where $\partial\theta$ is whatever the backward pass emits.  This is
not an implementation convention but the structural update rule of the
CRDC.  Whether the step is gradient \emph{descent} or
\emph{ascent}---and at what scale---is determined entirely by the sign
and magnitude of $\partial\theta$, which the learning-rate cap
(Section~\ref{sec:lrcap}) controls by injecting a positive or negative
seed into the backward pass.  No modification to the update rule itself
is required to switch between the two modes.

\section{Related Work}
\label{sec:relatedwork}

\subsection*{Cruttwell et al., 2022}

The categorical foundations of the framework are due to Cruttwell
et al.~\citep{cruttwell2022}, who showed that gradient-based learning
is naturally modelled as composition in a category $\mathbf{Para}(C)$
of parametric morphisms over a CRDC $C$.  They define parametric
lenses, the composition rule $(-.-)$, and the connection between
backpropagation and the CRDC reverse derivative, and demonstrate the
framework on small MLP examples.

The accompanying implementation is intentionally minimal: it is
dynamically typed, operates on single examples rather than batches, and
covers only dense layers.  The present work instantiates the same
categorical structure in Haskell, contributing: statically typed tensor
shapes via DataKinds; device and dtype polymorphism; batching as a
type-level parameter; convolutional and pooling architectures; and
zero-overhead evidence via GHC Core.  These contributions are discussed
in detail in Chapter~\ref{chap:casestudies}.

\subsection*{Backprop as a Functor}

Fong et al.~\citep{fong2019backpropfunctor} independently proposed
viewing backpropagation as a functor between categories of learners.
Their work establishes similar compositionality results for supervised
learning but takes a different categorical approach: composition of
learners corresponds directly to composition of update rules, without
the explicit Para construction.  Cruttwell et al.\ unify and generalise
both lines of work within the CRDC framework.

\subsection*{Conventional Frameworks: PyTorch and JAX}

The dominant operational frameworks treat neural networks as imperative
programs annotated with automatic differentiation.  Gradients are
computed by tracing the forward computation graph at runtime and
applying reverse-mode AD.

This differs from the categorical approach in three respects.  First,
the optimiser is a global object that maintains state separately from
model parameters; mixing update rules across layers requires explicit
parameter grouping.  In this framework the optimiser state is embedded
into the parameter type via |repara|, and per-layer optimisers arise
automatically from the product structure of |(.#.)|.  Second, tensor
shape errors typically surface at runtime when tensors are first
combined; the present framework reports them as type errors at compile
time.  Third, there is no clear categorical account of what the
composition of two PyTorch modules means mathematically; here,
composition is literally morphism composition in $\mathbf{Para}(C)$.

\subsection*{Hasktorch}

Hasktorch is the underlying tensor library used in this implementation.
It provides Haskell bindings to LibTorch (the C++ backend of PyTorch)
with a typed interface: |T.Tensor dv dt shape| tracks the device,
dtype, and shape at the type level, supplying the primitive operations
(matrix multiply, convolution, activation functions) that the framework
layers on top of.  Hasktorch is not itself a categorical ML framework;
it provides the typed computational substrate.
