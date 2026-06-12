%include polycode.fmt

%% ── Format directives ────────────────────────────────────────────────────────
%format ->    = "\to"
%format =>    = "\Rightarrow"
%format forall = "\forall"
%format .#.   = "\mathbin{\bullet}"
%format @     = "\mathbin{@}"
%% ─────────────────────────────────────────────────────────────────────────────

%if False
\begin{code}
module Appendix where
\end{code}
%endif

\chapter{Hasktorch Typed-API Bug Workarounds}
\label{app:bugfixes}

Two bugs in the hasktorch typed API (version 0.2.0.0, LibTorch~2.x) affect
the convolutional backward pass implemented in \texttt{Static/Layers.hs}.
Both were discovered during development of the |convLens| backward pass and
are worked around in \texttt{Static/Bugfix.hs}.

\section*{A.1\quad Transposed Convolution: Swapped \texttt{ConvSideCheck} Arguments}
\label{app:convtranspose}

The hasktorch typed binding for |convTranspose2d| applies the
|ConvSideCheck| constraint with the input and output spatial dimensions
swapped relative to the ATen semantics.  The library checks

\[
  \mathtt{ConvSideCheck}\;\mathit{outputSize}\;\mathit{kernelSize}\;\mathit{stride}\;\mathit{padding}\;\mathit{inputSize}
\]

\noindent where the correct direction is from input to output:

\[
  \mathtt{ConvSideCheck}\;\mathit{inputSize}\;\mathit{kernelSize}\;\mathit{stride}\;\mathit{padding}\;\mathit{outputSize}.
\]

\noindent For $1\!\times\!1$ kernels with unit stride and zero padding, the
two checks happen to be equivalent (the constraint reduces to
$\mathit{outputSize} = \mathit{inputSize}$ in both cases), so the bug is
invisible at that configuration.  The hasktorch test suite exercises only
$1\!\times\!1$ kernels for |convTranspose2d|, which is why the error was not
caught.  For all other kernel sizes the swapped constraint rejects valid calls
and accepts invalid ones.

The corrected binding in |Bugfix.hs| swaps the size arguments to
|ConvSideCheck| to match the ATen semantics, and was verified by
cross-checking output shapes against dynamic (untyped) calls on $3\!\times\!3$
and $4\!\times\!4$ kernels.

\section*{A.2\quad \texttt{im2col}: Unimplemented \texttt{Castable} Instance}
\label{app:im2col}

The hasktorch typed API does expose |im2col|, but its binding passes
Haskell tuples where the underlying ATen FFI expects lists.  The call
typechecks because a\newline |Castable (Int, Int) (ForeignPtr IntArray)| instance exists
\footnote{\url{https://github.com/hasktorch/hasktorch/blob/043552377932093d36e10d46e8b45aaa44225f2d/libtorch-ffi/src/Torch/Internal/Cast.hs\#L237}},
but that instance delegates to |makeTuple2|, which is defined as:
\footnote{\url{https://github.com/hasktorch/hasktorch/blob/043552377932093d36e10d46e8b45aaa44225f2d/libtorch-ffi/src/Torch/Internal/Class.hs\#L23}}

\begin{verbatim}
  makeTuple2 = error "makeTuple2 is not implemented."
\end{verbatim}

\noindent Any call to the hasktorch |im2col| binding therefore compiles
successfully but throws a Haskell runtime error unconditionally.

The corrected binding bypasses the original |Castable (Int, Int) (ForeignPtr IntArray)| by
converting each pair to a two-element list before the cast, and thus using\newline 
|Castable [Int] (ForeignPtr IntArray)|:

\begin{code}
im2col
  ::  U.Tensor -> (Int, Int) -> (Int, Int) -> (Int, Int) -> (Int, Int)
  ->  U.Tensor
im2col self (kh,kw) (dh,dw) (ph,pw) (sh,sw) =
  unsafePerformIO $ (cast5 im2col_tllll) self [kh,kw] [dh,dw] [ph,pw] [sh,sw]
\end{code}

\noindent The fix is otherwise identical to the original binding.  The
correctness of the output shape |[batch, inC * kH * kW, oH * oW]| was
verified against reference outputs on several kernel, padding, and stride
configurations.
