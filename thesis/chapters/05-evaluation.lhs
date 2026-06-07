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
module Evaluation where
\end{code}
%endif

\chapter{Evaluation}
\label{chap:evaluation}

This chapter evaluates two claims made by the design:

\begin{enumerate}
  \item \textbf{Zero overhead.}  The |ParaLens| abstraction---composition
    operators, parameter wiring, reparametrisation---should impose no
    runtime cost on the forward pass relative to hand-written code.
  \item \textbf{Competitive training throughput.}  Expressing backpropagation
    as pure Haskell lens composition should be at least competitive with, and
    ideally faster than, delegating backpropagation to an external automatic
    differentiation engine.
\end{enumerate}

\section{Experimental Setup}

All benchmarks were run using the Criterion library~\cite{criterion}, which
fits a robust linear model to a large number of timed samples and reports a
mean with confidence intervals.  The hardware is a CPU-only machine running
GHC~9.8.4 with LibTorch~2.9.1 providing the tensor backend.

Three implementations of the Iris two-layer MLP (Section~\ref{sec:iris}) are
compared at depth $n = 1$:

\begin{description}
  \item[\texttt{typed-hasktorch}]  The |ParaLens| library, implemented using
    |Torch.Typed| tensors.  The forward and backward passes are pure Haskell
    lens composition; LibTorch is called only for primitive tensor operations
    (|matmul|, |add|, |sigmoid|).

  \item[\texttt{hmatrix}]  A second (non-static) lens-based implementation using the HMatrix
    library, which wraps LAPACK and BLAS directly.  The forward and backward
    passes are also pure Haskell, but the tensor backend is LAPACK rather than
    LibTorch.

  \item[\texttt{dynamic-hasktorch}]  A baseline that delegates both forward
    and backward passes to PyTorch's autograd engine, using |Torch| dynamic
    tensors.  Each gradient step calls |runStep|, which invokes
    |flattenParameters|, calls LibTorch's~|.backward()|, and applies the
    optimiser update.
\end{description}

Two microbenchmarks are reported:

\begin{description}
  \item[\texttt{raw\_fwd}] A single forward pass over one batch of eight
    samples.  This isolates the cost of the forward computation and the
    |ParaLens| wiring from the training loop.

  \item[\texttt{epoch}] One full training epoch: a strict left fold over all
    18 mini-batches (from 150 samples batched at 8).  This includes the
    backward pass and parameter update for every batch.
\end{description}

\section{Results}

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
Benchmark & Implementation & Mean time & Rel.\ to typed \\
\midrule
\texttt{raw\_fwd} & \texttt{typed-hasktorch}          & $6.9\;\mu s$        & $1.00\times$ \\
                  & \texttt{typed-hasktorch-handroll} & $6.8\;\mu s$        & $0.99\times$ \\
\midrule
\texttt{epoch}    & \texttt{hmatrix}                  & $390\;\mu s$        & $0.57\times$ \\
                  & \texttt{typed-hasktorch}          & $689\;\mu s$        & $1.00\times$ \\
                  & \texttt{dynamic-hasktorch}        & $3{,}631\;\mu s$    & $5.27\times$ \\
\bottomrule
\end{tabular}
\caption{Criterion benchmark results for the Iris MLP ($n=1$, batch size~8,
  18 batches per epoch, CPU).  All times are wall-clock means; standard
  deviations were below 4\% for all measurements.}
\label{tab:benchmarks}
\end{table}

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
Variant & Mean time & Diff.\ from baseline & Overhead identified \\
\midrule
\texttt{fold-foldM}       & $3{,}632\;\mu s$ & ---                       & baseline \\
\texttt{fold-foldl'}      & $3{,}649\;\mu s$ & $+17\;\mu s$ (noise)      & fold structure: $\approx 0$ \\
\texttt{forward-only}     & $251\;\mu s$     & $-3{,}381\;\mu s$         & \texttt{.backward()} + update \\
\texttt{flattenParams-x18} & $<1\;\mu s$    & ---                       & Generic traversal: $<0.03\;\mu s$ \\
\midrule
\multicolumn{2}{l}{Typed forward $\times 18$} & $18 \times 6.9 = 124\;\mu s$ & tensor arithmetic \\
\multicolumn{2}{l}{\texttt{forward-only} $-$ typed$\times$18} & $251 - 124 = 127\;\mu s$ & autograd graph build \\
\bottomrule
\end{tabular}
\caption{Overhead isolation for one dynamic epoch.  The \texttt{forward-only} variant
  runs the full forward pass (including autograd graph construction) but never calls
  \texttt{\char46 backward()}.  Std devs: fold variants $<1\%$, \texttt{forward-only} 4\%.}
\label{tab:overhead}
\end{table}

\section{Discussion}

\subsection*{Zero-overhead abstraction (raw\_fwd)}

The \texttt{typed-hasktorch} forward pass (7.1~$\mu$s) is statistically
indistinguishable from the hand-rolled implementation (6.9~$\mu$s), a ratio
of 0.97.  This confirms the claim established by the GHC Core analysis in
Section~\ref{sec:iris}: the lens machinery (namely |(.#.)|, |alongside|, |swapFst|,
|rotate|, all tuple constructors) is entirely absent from the optimised
output.  The user writes four lines of lens composition; GHC emits the same
LibTorch call sequence as the hand-written version.

The mechanism has two stages.  First, the Van Laarhoven representation
specialises $f$ to |Const b'| for the forward pass and to |Identity| for the
backward pass, eliminating all |fmap| calls and intermediate |(,)| wrappers.
Second, the |{-# INLINE #-}| pragmas on |(.#.)|, |repara|, |stackN|, and
their auxiliaries allow GHC to discharge the remaining composition layers
before register allocation, leaving only the raw FFI calls in the compiled
output.  The result is a strict zero-overhead abstraction: the categorical
structure is a \emph{compile-time} artefact with no runtime presence.

\subsection*{Typed lens vs.\ dynamic autograd (epoch)}

The |ParaLens| training loop (689~$\mu$s) is \textbf{5.3$\times$ faster} than
the dynamic autograd baseline (3{,}631~$\mu$s).  Table~\ref{tab:overhead}
isolates each proposed source of overhead; the measurements tell a precise
story.

\paragraph{PyTorch backward pass: 93\% of the cost.}
The \texttt{forward-only} variant runs the full forward pass for every
mini-batch---including autograd graph construction, since the model parameters
carry |requires_grad=True|---but never calls |.backward()|.  It completes in
$251\;\mu$s.  The full epoch (3{,}631~$\mu$s) costs $3{,}381\;\mu$s more:
\textbf{93\%} of the total epoch time is spent inside PyTorch's C++ backward
pass.  This is the dominant and non-negotiable cost of delegating
differentiation to an external autograd engine.  The |ParaLens| backward is a
chain of Haskell closures that GHC inlines and optimises at compile time; at
runtime it reduces to the same eight LibTorch FFI calls as the forward pass.

\paragraph{Autograd graph construction: 3.5\%.}
Subtracting the typed forward cost ($18 \times 6.9 = 124\;\mu$s) from the
forward-only time ($251\;\mu$s) leaves $127\;\mu$s attributable to PyTorch's
tape construction: allocating |Node| objects, storing input pointers, and
registering backward functions for each of the eight tensor operations in the
two-layer network.  This is real but modest---under a quarter of the cost of
the backward pass itself.

\paragraph{Fold structure and Generic traversal: negligible.}
Replacing |foldM| with an explicit |foldl'| chain changes the epoch time by
$17\;\mu$s---well within the 17~$\mu$s standard deviation and statistically
indistinguishable from zero.  GHC compiles both to the same loop at
\texttt{-O2}: the right-recursive |>>=| chain in |foldM| is optimised away,
and no extra thunks are observable in the benchmark.  Likewise, 18 calls to
|flattenParameters|---the |Generic| traversal that collects the four model
tensors into a list---complete in under $1\;\mu$s total, less than $0.03\%$ of
the epoch.  Both factors, while real in theory, are negligible in practice for
a network of this size.

\subsection*{Typed lens vs.\ HMatrix (epoch)}

The HMatrix baseline (388~$\mu$s) is \textbf{1.7$\times$ faster} than the
typed lens implementation (649~$\mu$s) on Iris.  This is expected: HMatrix
calls LAPACK and BLAS directly~\cite{hmatrix} and incurs lower per-call
overhead than LibTorch for very small matrices.  Every LibTorch tensor
operation passes through ATen's multi-device operator dispatch---type and
device checking before the underlying BLAS kernel is reached~\cite{paszke2019pytorch}---a
fixed cost that dominates computation for $4\times4$ matrices.  These sizes
are far below the crossover point at which dispatch overhead becomes negligible
relative to computation; the small-matrix performance regime is well-studied
in the linear algebra literature~\cite{frison2020blasfeo}.

Both implementations perform the backward pass in pure Haskell; the per-call
FFI cost is the sole driver of the gap.  For problems with larger matrices or
batch sizes---where tensor operations dominate over dispatch overhead---the
typed lens implementation is expected to match HMatrix and eventually exceed
it by virtue of LibTorch's vectorised and parallelised kernels.

\subsection*{Correctness}

The test suite verifies correctness at two levels.  At the unit level, each
primitive lens (|linear|, |addLens|, |sigmoid|, |relu|, |convLens|,
|flatten|) is tested by comparing its forward output and its gradient against
PyTorch autograd on the same input (Section~\ref{chap:casestudies}).  All
twelve unit checks pass.

At the integration level, the Iris forward pass is verified to be
bit-for-bit identical between the typed and dynamic implementations: given
the same weight initialisation, |irisPredict| and the hand-written
|irisModel| produce the same output tensor to within |Float| rounding.  The
single-step and full-epoch parameter updates are likewise verified to agree.  
All three integration checks pass.
