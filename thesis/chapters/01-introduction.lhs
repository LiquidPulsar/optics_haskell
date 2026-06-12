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
The power of this formulation lies in its compositionality. Composing lenses corresponds to the physical ``wiring'' of sub-modules within a model. When we compose two parametric lenses, we are effectively defining how gradients should flow across the boundaries of different layers or loss functions. Crucially, the author of each lens writes only its \emph{local} backward pass; composition assembles these local passes into the global chain rule automatically, and the combined parameter type is inferred rather than declared. This is what distinguishes the approach from hand-rolled backpropagation, where the wiring between layers must itself be written and maintained by hand.

This approach offers significant advantages over standard modularity:
\begin{itemize}
    \item \textbf{Recursive Scaling:} Large-scale architectures can be treated as single lenses composed of smaller, verified lenses, allowing for a rigorous analysis of complex systems through their constituent parts. 
    \item \textbf{Internalised Optimisation and Deep Compositionality:} In standard practice, an optimiser is an external control loop that sits outside the model. In this framework, an optimiser is reinterpreted as a reparameterisation lens. This allows for ``Deep Compositionality'': one can easily ``wire'' different optimisers into different sub-modules of a single architecture (e.g., using an Adam-lens for convolutional layers and an SGD-lens for dense layers). The resulting hybrid structure is itself a single lens, abstracting away internal complexity and presenting a uniform interface to the rest of the system.
\end{itemize}


\subsection{The Cost of Dynamic Differentiation}

Reverse-mode automatic differentiation, as provided by frameworks such as
PyTorch, makes this convenient: the user writes only the forward pass, and
the backward pass is built automatically by recording each operation onto a
\emph{tape} and replaying it in reverse. The convenience has a structural
cost. The tape is rebuilt on every step, so each pass pays for graph-node
allocation and dynamic dispatch, and shapes are checked only when a kernel
runs, so an architectural mismatch surfaces as a runtime exception rather
than a rejected program. The parametric lens framework sits at the opposite
end: each lens carries its own backward pass, so there is no tape and the
chain rule is discharged structurally by composition, but the cost is paid
up front by the library author, who must derive every backward pass by
hand. The central question of this thesis is whether a sufficiently
expressive host language can make that trade pay: whether the hand-derived
passes can be \emph{verified}, so that surrendering automatic
differentiation introduces no silent gradient bugs, and \emph{compiled
away}, so that the abstraction imposes no runtime penalty over hand-written
code.

\subsection{Implementation in Haskell}

The objective of this project is to instantiate the Cruttwell et al.\
framework in Haskell.  Previous implementations have used Python
\cite{catlearning}; Haskell offers a strong type system and mature
libraries for optics \cite{Pickering_2017, ekmett2025lens} that make the
type-level guarantees central to this thesis expressible directly in the
language.  Two features of the language do the decisive work.  The van
Laarhoven encoding of lenses (Chapter~\ref{chap:background}) represents a
lens as a single higher-rank function, so that composition is ordinary
function composition and the whole abstraction collapses to direct calls
once the compiler fixes the functor at each use site; this is what makes
the zero-overhead result attainable.  And PyTorch's autograd, rather than
being a runtime dependency, is repurposed as a \emph{testing oracle}: every
hand-derived backward pass is cross-validated against it to a tolerance of
$10^{-4}$, so the gradients are written by hand but checked by machine.

This thesis asks: can the Cruttwell et al.\ framework be
realised in Haskell with zero abstraction overhead and static
architectural safety, and at what cost to expressiveness?  The question is
one of foundations rather than scale: the aim is to establish that the
categorical structure can be carried into a real language with its
guarantees intact, and to map where those guarantees end.  The library is
accordingly built against representative models (classifiers, an
autoencoder, attention) rather than production-scale ones, and
Chapter~\ref{chap:conclusion} marks the boundary of the current design:
the stateful and stochastic machinery (dropout, batch normalisation,
global optimiser state) that the lens model does not yet reach.

\section{Contributions}
\label{sec:contributions}

This thesis makes the following original contributions relative to
the existing literature, and in particular to the Python
proof-of-concept that accompanies Cruttwell et al.~\cite{catlearning}:

\begin{enumerate}

  \item \textbf{Static type safety for tensor computation.}
    Tensor shapes, batch sizes, devices, and dtypes are encoded as
    type-level parameters via \texttt{DataKinds} and constraint synonyms,
    so shape, batch, and device/dtype mismatches are compile-time errors;
    the original Python implementation performs no such checking.

  \item \textbf{Type-level variable network depth.}
    The \texttt{Stack.hs} module uses Peano naturals and typeclass
    induction to build networks of depth $n$ whose parameter type is a
    fully concrete nested tuple at each $n$, preserving GHC's ability
    to specialise and unbox the entire parameter structure with zero
    overhead relative to a hand-written network of the same depth.

  \item \textbf{Extended architecture support.}
    The framework reaches beyond the MLP scope of the original paper:
    autoencoders are ordinary |ParaLens'| pipelines composed end-to-end
    via |(.#.)| with the bottleneck enforced at the type level, and
    residual skip connections are a single higher-order combinator
    |skipPara| built from existing primitives, with the identity gradient
    path emerging automatically from the combinator structure.

  \item \textbf{Attention as a \texttt{ParaLens'}.}
    Scaled dot-product self-attention and its multi-head generalisation
    are implemented with fully hand-derived backward passes, including
    the rank-one Jacobian correction for the softmax non-linearity.
    The constraint $e = h \cdot \mathit{hd}$ is enforced at the type
    level, so a head-count or embedding-dimension mismatch is a
    compile-time error rather than a silent shape failure at runtime.

  \item \textbf{Zero-overhead evidence via GHC Core.}
    Compiling the lens-based forward pass and an equivalent hand-written
    one with \texttt{-O2 -ddump-simpl} yields structurally identical
    worker functions: the entire \texttt{ParaLens} abstraction
    (composition, reparametrisation, the van Laarhoven \texttt{forall})
    is absent from the optimised output.

\end{enumerate}

\section{Thesis Outline}

Chapter~\ref{chap:background} covers the background: Haskell's type
system, optics (including the van Laarhoven representation),
gradient-based learning, and Cartesian Reverse Differential Categories.
Chapter~\ref{chap:core} defines the \texttt{ParaLens} type, its
composition operator (|.#.|), and the \texttt{repara} combinator.
Chapters~\ref{chap:layers}, \ref{chap:loss}, and \ref{chap:optim} cover
layers, loss functions, and optimisers; Chapter~\ref{chap:casestudies}
presents the Iris and MNIST case studies, the GHC Core zero-overhead
evidence, and the type-level depth experiment; and
Chapter~\ref{chap:conclusion} concludes.

\paragraph{Source code.}
The full implementation is available at
\url{https://github.com/LiquidPulsar/optics_haskell/tree/static}.