%include polycode.fmt

%if False
\begin{code}
module Conclusion where
\end{code}
%endif

\chapter{Conclusion}
\label{chap:conclusion}

\section{Summary}

This thesis has instantiated the categorical framework of Cruttwell
et al.~\citep{cruttwell2022} as a working Haskell library for
gradient-based machine learning.  The central idea is that every
component of a training pipeline---layers, loss functions, optimisers,
learning-rate schedules---is a parametric lens: a bidirectional map
that carries a forward pass and a backward (gradient) pass as a single
composable unit.  Sequential composition via \texttt{(.{}\#{}.)} wires
layers into networks while automatically accumulating their parameter
types into a product; the type system then ensures that the combined
parameter is always correctly structured.

The library extends the original framework in seven concrete ways
(Section~\ref{sec:contributions}).  Tensor shapes, batch sizes, devices,
and dtypes are all static: a misconfigured architecture produces a
compile-time type error rather than a runtime crash.  Convolutional and
pooling layers extend the MLP-only scope of the Cruttwell paper.
Variable-depth networks are expressed via type-level Peano naturals in
\texttt{Stack.hs}, and GHC's inliner reduces the abstraction to the
same machine code as a hand-written equivalent.  Per-layer optimisers
follow automatically from the product structure of
\texttt{(.{}\#{}.)} without any global optimiser state.

The Iris and MNIST case studies demonstrate these properties concretely.
The GHC Core comparison for the Iris model shows that the
\texttt{ParaLens} abstraction is entirely absent from the optimised
output: both the lens-based and hand-rolled forward passes reduce to the
same eight tensor operations.  MNIST shows the framework scaling to a
convolutional architecture with statically inferred intermediate shapes
and He-initialised kernels.

\section{Future Work}

Several directions remain open for future investigation.

\paragraph{Attention mechanisms.}
The self-attention operation $\mathrm{Attn}(x) = \mathrm{softmax}(xW_Q
(xW_K)^\top / \sqrt{d}) \cdot xW_V$ is an instance of a differentiable
function and could in principle be expressed as a \texttt{ParaLens'} with
parameter type $(W_Q, W_K, W_V, W_O)$.  The challenge is the fan-out on
$x$: the query and key projections both depend on the same input, which
requires a diagonal morphism (duplicating $x$ across two parallel
\texttt{matMulLens} calls) before combining them via the scaled
dot-product.  This is structurally supported by \texttt{alongside} and
would extend the framework to transformer architectures, which are
currently the dominant model class in natural language processing and
vision.

\paragraph{Recurrent architectures.}
Recurrent networks maintain hidden state across time steps, which does
not fit the stateless composition pattern of \texttt{(.{}\#{}.)}
directly.  One avenue is to treat the hidden state as an additional
parameter, making the recurrence an instance of \texttt{repara} on the
state wire.  Whether this recovers standard BPTT or requires a
categorical extension is an open question.

\paragraph{Performance benchmarks.}
The performance table in Chapter~\ref{chap:casestudies}
(Table~\ref{tab:iris-perf}) remains to be filled with timing results
for single- and double-precision training.  A systematic comparison of
the framework's wall-clock training time against equivalent PyTorch
code would quantify the overhead (if any) of the Haskell/LibTorch FFI
boundary, separate from the abstraction overhead already shown to be
zero by the GHC Core analysis.

\paragraph{Formal verification of lens laws.}
The lens laws (get-put, put-get, put-put) are satisfied structurally
by construction in this framework, but are not machine-verified.
Property-based testing with QuickCheck could confirm the laws hold for
all implemented lenses under arbitrary inputs, providing stronger
assurance than the informal argument from construction.

\paragraph{Discrete and probabilistic CRDCs.}
The Cruttwell et al.\ framework is defined for any CRDC, not just
smooth Euclidean spaces.  Instantiating the Haskell library for a
discrete CRDC (e.g., a category of boolean circuits) or a
probabilistic one would generalise gradient-based learning beyond the
real-valued setting and test the abstraction boundaries of the current
design.
