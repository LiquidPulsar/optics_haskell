%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%format ->    = "\to"
%format =>    = "\Rightarrow"
%format forall = "\forall"
%format .#.   = "\mathbin{\bullet}"
%format ***   = "\mathbin{\times}"
%format <$>   = "\mathbin{\langle\$\rangle}"
%% ─────────────────────────────────────────────────────────────────────────────

\chapter{Introduction}

\section{The Need for a Unifying Perspective}

The dominant paradigm in modern artificial intelligence, particularly deep learning, relies heavily on gradient-based optimisation techniques \cite{lecun2015deep}. While these methods have achieved remarkable success across various scientific and technological domains, the ever-increasing complexity of both models and their underpinning algorithms come hand in hand with an increasing need for explainability and rigorous analysis. Current interpretations typically rely on heuristic combinations of distinct algorithmic components, often lacking a thorough, unified mathematical foundation that can scale with the sophistication of modern models \cite{shiebler2021categorytheorymachinelearning}.



% The overwhelming majority of modern machine learning is built via Python glue code stitching together calls to C libraries and GPU kernels. This approach is difficult to optimise fully, difficult to reason with, and is generally not typesafe.


\subsection{Structural Fragmentation}

A typical supervised learning scenario orchestrates an interaction between three seemingly distinct entities: the model, the loss map, and the optimiser \cite{Plaut1986ExperimentsOL}. In standard practice, these components are treated as independent modules: one might swap Softmax cross-entropy for focal loss, or replace basic gradient descent with adaptive methods like Adam or Nesterov momentum \cite{catlearning, fong2019backpropfunctor}.

However, this modularity is often implemented operationally rather than mathematically. Questions regarding the shared structural properties of these components, and whether they can be described by a single uniform language, remain largely unaddressed in standard frameworks. Furthermore, standard approaches usually restrict learning to continuous domains (smooth maps), often failing to generalise to discrete settings such as Boolean circuits \cite{catlearning}.

\subsection{Parametric Lenses}

This project explores a unifying semantic framework proposed by Cruttwell et al. \cite{catlearning}, which posits that gradient-based learning is fundamentally a composition of parametric lenses. In this view, a lens serves as a formal interface between internal parameters and external observables. It provides a mechanism to ``zoom'' into specific components of a model, adjust their internal states, and propagate those changes back through the system.

This perspective abstracts the learning process into three core principles:
\begin{enumerate}
    \item \textbf{Parameterisation:} All components—from neural weights to loss functions—are viewed as parameterised maps \cite{catlearning}. This lifts standard functions into a space where their internal configurations are first-class citizens.
    \item \textbf{Bidirectional Information Flow:} The architecture must support both the forward propagation of predictions and the backward propagation of updates. This is captured by the categorical structure of lenses \cite{fong2019lensess}, where the ``view'' (forward pass) and ``update'' (backward pass) are intrinsically linked.
    \item \textbf{Abstract Differentiation:} Parameter updates are driven by differentiation, modeled via Cartesian Reverse Differential Categories (CRDCs) \cite{elliott2018simpleautodiff}.
\end{enumerate}

By viewing the entire training loop as a morphism in a structured category, we move away from ``black box'' optimisation. Instead, the optimiser is reinterpreted not as an external control loop, but as a specific type of lens that reparameterises the model, creating a clean algebraic structure for gradient flows.

\subsection{Modularity through Composition}
The power of this formulation lies in its compositionality. Composing lenses corresponds to the physical ``wiring'' of sub-modules within a model. When we compose two parametric lenses, we are effectively defining how gradients should flow across the boundaries of different layers or loss functions.

This approach offers significant advantages over standard modularity:
\begin{itemize}
    \item \textbf{Recursive Scaling:} Large-scale architectures can be treated as single lenses composed of smaller, verified lenses, allowing for a rigorous analysis of complex systems through their constituent parts. 
    \item \textbf{Internalised Optimisation and Deep Compositionality:} In standard practice, an optimiser is an external control loop that sits outside the model. In this framework, an optimiser is reinterpreted as a reparameterisation lens. This allows for ``Deep Compositionality'': one can easily ``wire'' different optimisers into different sub-modules of a single architecture (e.g., using an Adam-lens for convolutional layers and an SGD-lens for dense layers). The resulting hybrid structure is itself a single lens, abstracting away internal complexity and presenting a uniform interface to the rest of the system.
\end{itemize}


\subsection{Implementation in Haskell}

The objective of this project is to instantiate the Cruttwell et al.\
framework in Haskell.  Previous implementations have used Python
\cite{catlearning}; Haskell offers a strong type system and mature
libraries for optics \cite{Pickering_2017, ekmett2025lens} that make the
type-level guarantees central to this thesis expressible directly in the
language.

\section{Contributions}
\label{sec:contributions}

This thesis makes the following original contributions relative to
the existing literature, and in particular to the Python
proof-of-concept that accompanies Cruttwell et al.~\cite{catlearning}:

\begin{enumerate}

  \item \textbf{Statically typed tensor shapes.}
    Tensor dimensions are encoded as type-level natural numbers via
    GHC's \texttt{DataKinds} extension.  A matrix multiply or
    convolution that receives a tensor of the wrong shape is rejected
    at compile time, not at runtime.  The original Python
    implementation performs no shape checking.

  \item \textbf{Batching as a type parameter.}
    The mini-batch size $b$ is a type-level \texttt{Nat}.  All model
    signatures are polymorphic in $b$; a mismatch between the batch
    size declared in the model type and the batch size of the input
    data is a compile-time type error.

  \item \textbf{Device and dtype polymorphism.}
    A single model definition is polymorphic over the compute device
    (CPU, CUDA) and element dtype (32-bit float, 64-bit double).
    Switching from CPU to GPU or from single to double precision
    requires only a change to the type application at the call site.
    Constraint synonyms prevent invalid device/dtype combinations from
    type-checking.

  % \item \textbf{Convolutional and pooling architectures.}
  %   The Cruttwell et al.\ framework is extended beyond dense layers to
  %   include convolutions and max-pooling.  Convolution output shapes
  %   (kernel size, stride, and padding are all type-level tuples) are
  %   inferred by type families, so architectural mismatches are
  %   compile-time errors.

  \item \textbf{Autoencoder architectures.}
    The framework is extended to autoencoder networks, where encoder and
    decoder are each ordinary |ParaLens'| pipelines that compose with
    |(.#.)| into a single end-to-end model.  The bottleneck constraint
    (a latent dimension smaller than the input) is enforced at the type
    level with no special-casing: the composition rule handles the
    dimension change identically to any other layer transition.  The
    model is trained on MNIST images with MSE reconstruction loss,
    demonstrating that the framework is not limited to discriminative tasks.

  \item \textbf{Skip connections as a combinator.}
    Residual shortcut connections are expressed as a single higher-order
    combinator |skipPara :: ParaLens' p a a -> ParaLens' p a a|, built
    entirely from existing primitives---|splitIso|, |from rotate|,
    |alongside|, and |from splitIso|---with no new lens axioms.
    The identity gradient path, which ensures gradients reach early
    layers even when the learned branch saturates, emerges automatically
    from the combinator structure: no separate backward-pass derivation
    is required.  Stacking |skipPara|-wrapped blocks via |stackN| yields
    a residual network whose full parameter type is inferred by the
    type system.

  \item \textbf{Attention as a \texttt{ParaLens'}.}
    Scaled dot-product self-attention and its multi-head generalisation are
    implemented as |ParaLens'| values with fully manually derived backward
    passes, including the rank-one Jacobian correction for the softmax
    non-linearity.  The multi-head variant enforces the constraint
    $e = h \cdot \mathit{hd}$ at the type level via GHC's equality
    constraints, so a head-count or embedding-dimension mismatch is a
    compile-time error rather than a silent shape failure at runtime.

  \item \textbf{Type-level variable network depth.}
    The \texttt{Stack.hs} module uses type-level Peano naturals and
    typeclass induction to build networks of depth $n$ whose parameter
    type is a fully concrete nested tuple at each $n$.  This preserves
    GHC's ability to specialise and unbox the entire parameter
    structure, yielding zero overhead relative to a hand-written
    network of the same depth.

  % \item \textbf{Per-layer optimisers via the product type.}
  %   Because the composition operator \texttt{(.{}\#{}.)} builds the
  %   combined parameter type as a product, different layers can carry
  %   different optimiser states (momentum buffer, gradient accumulator,
  %   Adam moments) with no global optimiser object and no explicit
  %   parameter grouping.  The type system enforces that each layer's
  %   update rule is local.

  \item \textbf{Zero-overhead evidence via GHC Core.}
    Compiling the lens-based forward pass and an equivalent
    hand-written forward pass with \texttt{-O2 -ddump-simpl} produces
    structurally identical worker functions.  The entire
    \texttt{ParaLens} abstraction---composition, reparametrisation,
    the van Laarhoven \texttt{forall}---is absent from the optimised
    output.

\end{enumerate}

\section{Thesis Outline}

Chapter~\ref{chap:background} covers the necessary background in
Haskell's type system, optics (including the van Laarhoven lens
representation), gradient-based learning, and Cartesian Reverse
Differential Categories.
Chapter~\ref{chap:core} defines the \texttt{ParaLens} type, its
composition operator (|.#.|) and the \texttt{repara}
combinator.
Chapters~\ref{chap:layers}, \ref{chap:loss}, and \ref{chap:optim}
describe the library's layers, loss functions, and optimisers
respectively.
Chapter~\ref{chap:casestudies} presents the Iris and MNIST case
studies, including the GHC Core zero-overhead evidence and the
type-level depth experiment.
Chapter~\ref{chap:conclusion} concludes and outlines directions for
future work.