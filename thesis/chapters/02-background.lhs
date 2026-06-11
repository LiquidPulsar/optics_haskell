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
Five advanced type-system features support the framework: type classes for
overloaded interfaces; the kind system and |DataKinds| for lifting tensor
shapes to the type level; type families for computing output shapes at
compile time; higher-rank polymorphism for composable lenses; and constraint
synonyms for keeping signatures readable.

\subsection*{Type Classes}

The primary abstraction mechanism is the \emph{type class}: a class
declares an interface and instances provide concrete implementations.
Type class resolution is purely compile-time---GHC selects and inlines
the appropriate instance, leaving no dispatch overhead at runtime.
This property is crucial in Section~\ref{sec:zero-overhead}, where
the entire lens abstraction is shown to vanish in the compiled output.

\subsection*{The Kind System and DataKinds}

Every type in Haskell has a \emph{kind}, which is the ``type of a
type.''  Ordinary types such as |Int| have kind |Type|.  A type
constructor such as |Maybe| has kind |Type -> Type|: it takes one type
argument before producing a concrete type.

The |{-# LANGUAGE DataKinds #-}| extension promotes value-level data
constructors to the type level.  The standard natural numbers, for
instance, are promoted to kind |Nat|, and lists of naturals to kind
|[Nat]|.  The typed tensor library Hasktorch exploits this directly: a tensor on
device |dv|, with element type |dt|, and shape |[m, n]| is written:

\begin{code}
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
compiler. \newline The |{-# LANGUAGE TypeFamilies #-}| extension enables
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

% \section{Optics}
% \label{sec:optics-bg}

% An \emph{optic} is a composable interface for accessing and modifying
% a sub-component of a data structure.  The simplest optic is a
% \emph{lens}, which focuses on a single sub-component.  Concretely, a
% lens is a getter paired with a setter:

% \begin{code}
% data Lens s a = Lens
%   { view :: s -> a
%   , set  :: s -> a -> s }
% \end{code}

% \noindent This representation is straightforward but does not compose
% directly: chaining two lenses requires manually threading the getter
% and setter, producing boilerplate that grows with nesting depth.
% Van Laarhoven~\citep{van2009lens} showed that encoding both operations
% as a single higher-rank function eliminates this problem---two lenses
% then compose by plain function composition |(.)| with no glue code.
% The central reason lenses appear throughout the framework is precisely
% this composability.  The full derivation, including the |Identity| and
% |Const| functor trick, the polymorphic generalisation, and the lens
% laws, is given in Section~\ref{sec:lenses}.

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
(CRDC)~\citep{catlearning}.  CRDCs provide an axiomatic account of
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
(Section~\ref{sec:losscaps}) controls by injecting a positive or negative
seed into the backward pass.  No modification to the update rule itself
is required to switch between the two modes.


%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%% Symbolic operators rendered as math
%format ->          = "\to"
%format =>          = "\Rightarrow"
%format forall      = "\forall"
%format <$>         = "\mathbin{\langle\$\rangle}"
%format `div`       = "\mathbin{\div}"

%% Named variables → Greek / math letters (use these names in code blocks)
%format phi         = "\phi"
%format alpha       = "\alpha"

%% Lens type operators
%format Lens'       = "\mathit{Lens}^{\prime}"
%% ─────────────────────────────────────────────────────────────────────────────

\section{Lenses and Optics}
\label{sec:lenses}

Compound data structures are ubiquitous in modern programming, and their
inherent modularity is key to building and reasoning with complex ensembles.
Lenses\cite{foster2005combinators} are one approach to solve the problem of
accessing the components of such structures, also known as the
\textit{view-update problem}\cite{updatesemantics}.

\subsection{Origins}

A simple lens from outer type $S$ to inner type $A$ (which we shall denote as
|Lens' s a| for reasons that will become apparent) can be thought of as
comprising\cite{foster2005combinators} a function |view :: s -> a| which
projects the component $A$, and a function |update :: s -> a -> s| which
accepts a new value of $A$ and uses it to modify a given $S$.

\begin{code}
data Lens' s a = Lens' {view :: s -> a,  update  :: s -> a -> s}
\end{code}

As a concrete example, consider the relationship between the following
|Account| and |PhoneNumber| data types:

\begin{code}
newtype PhoneNumber  = PhoneNumber Int
newtype Account      = Account
    {  phone      :: PhoneNumber
    ,  accountId  :: Int
    }

acctToPhone :: Lens' Account PhoneNumber
acctToPhone = Lens' view update
  where
    view    :: Account -> PhoneNumber
    view    = phone

    update  :: Account -> PhoneNumber -> Account
    update acct num = acct { phone = num }
\end{code}

\subsection{Van Laarhoven Lenses}
\label{sec:optics-bg}

It is tempting to extend our |Lens| type with a modification function
|mod :: (a -> a) -> s -> s|\cite{peytonjones2013lenses}. This begs the
question: at what point do we stop? We could usefully include
|modM :: (a -> Maybe a) -> s -> Maybe s|, or even
|modIO :: (a -> IO a) -> s -> IO s|. Before letting this get out of hand, we
observe that both share the structure of a |Functor|, yielding
|modF :: Functor f => (a -> f a) -> s -> f s|.

As a quick refresher, the category of |Functor|s\cite{haskell2010,
haskell-base-functor} supports the lifting of an arbitrary function into their
type. In Haskell terms:

\begin{code}
class Functor f where
    fmap :: (a -> b) -> f a -> f b
\end{code}

By careful choice of specific |Functor|s we can recover |view|, |mod|, and
|update|. Note that providing |const a| to |mod| recovers |update|, so we need
only produce the first two. The key choices are~\cite{ghc_internal_identity,
ghc_internal_const}:

\begin{code}
newtype Identity  a    = Identity  {  runIdentity  :: a    }
newtype Const     a b  = Const     {  getConst     :: a    }

instance Functor Identity where
    fmap f (Identity x) = Identity (f x)

instance Functor (Const m) where
    fmap _ (Const v) = Const v
\end{code}

Applying |Identity| to |modF| yields |modF :: (a -> Identity a) -> s -> Identity s|,
isomorphic to |mod|---all that is left is wrapping and unwrapping the
|Identity|\footnote{This has no runtime cost thanks to the semantics of
\texttt{newtype} and \texttt{coerce}\cite{breitner2014safe}.}.

Applying |Const a| to |modF| yields
|modF :: (a -> Const a a) -> s -> Const a s|,
isomorphic to |(a -> a) -> s -> a|, which when provided |id| is exactly |view|.

Thus a |Lens'| only needs |modF|, and we can simplify to the type alias:

\begin{code}
type Lens' s a = forall f . Functor f => (a -> f a) -> s -> f s
\end{code}

This representation also lets GHC apply aggressive optimisations for
higher-order functions, which will matter as we compose increasingly complex
lenses.

It is often convenient to allow the types in the reverse direction to differ
from those in the forward direction, yielding
|update :: s -> b -> t| for fresh types $B$ and $T$:

\begin{code}
data Lens s t a b = Lens {view :: s -> a,  update  :: s -> b -> t}
\end{code}

In the Van Laarhoven style~\cite{ekmett2025lens}:

\begin{code}
type Lens s t a b  = forall f . Functor f => (a -> f b) -> s -> f t
type Lens' s a     = Lens s s a a

lens :: (s -> a) -> (s -> b -> t) -> Lens s t a b
-- Note: fmap written infix
lens sa sbt afb s = sbt s <$> afb (sa s)
\end{code}

\subsection{Composition and Modularity}

The Van Laarhoven encoding gives composition for free. Given:

\begin{code}
x  :: Lens s t a b  -- forall f . Functor f => (a -> f b) -> (s -> f t)
y  :: Lens a b c d  -- forall f . Functor f => (c -> f d) -> (a -> f b)
\end{code}

Standard function composition |(.)| has type |(b -> c) -> (a -> b) -> (a -> c)|,
so |x . y| unifies immediately:

\begin{code}
-- x . y :: Lens s t c d
--        = forall f . Functor f => (c -> f d) -> (s -> f t)
\end{code}

No glue code required: composing lenses with |(.)| is simply function
composition. This allows compound accessors to be built from primitives without
any overhead.

\subsection{Final Note}

So far we have considered only lenses whose focus is a concrete field of a
record, as with |PhoneNumber| inside |Account|. This is not a requirement, and
in subsequent sections we will see that machine learning layers can themselves
be expressed as more exotic lenses.



\section{Related Work}
\label{sec:relatedwork}

\subsection*{Cruttwell et al., 2022}

The categorical foundations of the framework are due to Cruttwell
et al.~\citep{catlearning}, who showed that gradient-based learning
is naturally modelled as composition in a category $\mathbf{Para}(C)$
of parametric morphisms over a CRDC $C$.  They define parametric
lenses, the composition rule, and the connection between
backpropagation and the CRDC reverse derivative, and demonstrate the
framework on small MLP examples.

The accompanying implementation is intentionally minimal: it is
dynamically typed, operates on single examples rather than batches, and
covers only dense layers and convolutions.  

Importantly, the implementation in Python (whilst a natural choice for ease of development) 
does not fulfil the original intentions of the lens design in terms of efficient composition. 
Python imposes significant overhead due to the additional function calls required for composing 
our lenses, which cannot be optimised out.

Haskell offers far better support for lenses: thanks to its optimising compiler and powerful type 
reasoning capabilities we can eliminate all overhead vs hand-written code.
The present work instantiates the prior categorical structure in Haskell, 
contributing: statically typed tensor
shapes via DataKinds; device and dtype polymorphism; batching as a
type-level parameter; autoencoders and attention mechanisms; 
type-level variable network depth and
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
