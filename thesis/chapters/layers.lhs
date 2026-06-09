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
module Layers where
\end{code}
%endif

\chapter{Layer Implementation}
\label{chap:layers}

Each concrete neural network layer is a term of type |ParaLens'| or |Lens'|,
directly instantiating the wire diagram of Section~\ref{sec:wirediagrams}.
The guiding principle is simple: \emph{stateless} transformations---those with no
learnable parameters---are plain |Lens'| values with no vertical parameter wires;
\emph{stateful} ones are |ParaLens'| values whose parameter type holds the
learnable tensors.  This distinction is enforced at the type level and is not a
matter of convention.

\section{Linear Layer and Bias}

The linear layer is the framework's core building block; its backward
pass establishes the gradient-flow pattern---getter computes the forward
output, setter returns gradients for both weights and inputs---that
every subsequent layer follows.

The weight matrix alone constitutes a |ParaLens'| whose parameter is a
typed tensor of shape |[o, i]|:

\begin{code}
linear  ::  T.MatMulDTypeIsValid dv dt
        =>  ParaLens'
              (Tensor dv dt [o, i])
              (Tensor dv dt [batch, i])
              (Tensor dv dt [batch, o])
linear = lens fwd rev
  where
    fwd (w, x)      = T.matmul x (transp w)
    rev (w, x) grad = (T.matmul (transp grad) x, T.matmul grad w)
\end{code}

\noindent The forward pass computes $xW^\top$; the backward pass applies the
standard matrix-calculus identities $\partial L/\partial W = \mathit{grad}^\top x$
and $\partial L/\partial x = \mathit{grad}\,W$.  The helper |transp| witnesses
the transposition at the type level by permuting the first two index positions.

Bias addition is a separate |ParaLens'| that broadcasts a vector across the batch
dimension and collapses it back on the reverse pass:

\begin{code}
addLens  ::  CanAddLens dv dt
         =>  ParaLens'
               (Tensor dv dt shape)
               (Tensor dv dt (b : shape))
               (Tensor dv dt (b : shape))
addLens = lens fwd rev
  where
    fwd = uncurry T.add
    rev _ = T.sumDim @0 &&& id
\end{code}

\noindent The gradient with respect to the bias is a sum over the batch axis
(|@0| passes the axis index to |sumDim|); the gradient with respect to the
input is the identity.  Composing the two
with |(.#.)| yields a full affine layer in a single line:

\begin{code}
matMulLensCore  ::  CanMMLens dv dt
                =>  ParaLens'
                      (Tensor dv dt [o, i], Tensor dv dt [o])
                      (Tensor dv dt [batch, i])
                      (Tensor dv dt [batch, o])
matMulLensCore = linear .#. addLens
\end{code}

\noindent The combined parameter |(Tensor [o,i], Tensor [o])| arises
automatically from the composition rule: the product parameter type requires
no manual construction.  The full backward pass---computing both $\partial
L/\partial W$, $\partial L/\partial b$, and $\partial L/\partial x$---is wired
together from the two constituent lenses without any additional code.

For architectures that require a linear map over arbitrary leading dimensions,
|linear'| relaxes the fixed |[batch, i]| constraint to any shape satisfying
|IsSuffixOf [i] shape|.  The typed tensor API does not expose this general
matmul rule, so the implementation wraps |U.matmul| with |UnsafeMkTensor|;
the constraint |KnownNat i| supplies the static evidence needed to reconstruct
the correct output shape at runtime.

\section{Activation Functions}

Activation functions carry no learnable parameters and are plain |Lens'| values:

\begin{code}
sigmoid  ::  T.StandardFloatingPointDTypeValidation dv dt
         =>  Lens' (Tensor dv dt shape) (Tensor dv dt shape)
sigmoid = lens T.sigmoid rev
  where
    rev = (*) . ap (*) (1 -) . T.sigmoid

relu     ::  T.StandardFloatingPointDTypeValidation dv dt
         =>  Lens' (Tensor dv dt shape) (Tensor dv dt shape)
relu = lens T.relu ((*) . heaviside)
\end{code}

\noindent The sigmoid backward pass uses $\sigma'(x) = \sigma(x)(1 - \sigma(x))$,
written point-free as |(*) . ap (*) (1 -) . T.sigmoid|: the incoming gradient is
multiplied element-wise by $\sigma(x)$ and $(1 - \sigma(x))$.  The ReLU backward
multiplies by the Heaviside step $H(x) = \mathbf{1}_{x > 0}$, realised as a
Boolean tensor cast to the output dtype:

\begin{code}
heaviside = T.toDType @dt @T.Bool . liftA2 ($) T.gt T.zerosLike
\end{code}

\noindent The |@dt| and |@T.Bool| type applications supply the target and
source dtype to |toDType|: the Boolean comparison mask produced by |T.gt|
is cast to the floating-point type |dt| that the rest of the pipeline uses.

\noindent Because these layers are plain |Lens'| values they carry no vertical
parameter wires in the wire diagram sense.  A |Lens'| can be appended to a
|ParaLens'| pipeline using ordinary Haskell function composition |(.)| rather
than the parametric |(.#.)|.  Since |ParaLens' p a b = Lens' (p, a) b|, the
output type of any |f :: ParaLens' p a b| is exactly the input type expected
by any |g :: Lens' b c|, so |f . g| type-checks directly as a
|ParaLens' p a c| with the parameter type |p| unchanged:

\begin{code}
affineRelu = matMulLensCore . relu
\end{code}

\noindent This has the same parameter type |(Tensor [o,i], Tensor [o])| as
|matMulLensCore| alone.  The alternative |matMulLensCore .#. toPara relu|
would also type-check but would introduce a spurious unit into the combined
parameter: |(Tensor [o,i], Tensor [o], ())|.

The operator precedence of |(.#.)| is set one lower than that of |(.)| by
design, so multi-layer stacks read cleanly without parentheses:

\begin{code}
net = matMulLensCore . relu .#. matMulLensCore . relu .#. matMulLensCore
\end{code}

\noindent This parses as
|(matMulLensCore . relu) .#. (matMulLensCore . relu) .#. matMulLensCore|:
each |(.#.)| joins two already-clean |ParaLens'| values, so the combined
parameter is a nested product of weight and bias tensors only, with no
unit components anywhere in the type.

\section{Convolution}

Convolution illustrates how \texttt{DataKinds}-indexed shapes extend
the compile-time safety guarantee beyond simple matrix sizes: the
kernel dimensions, input spatial size, stride, and padding must all be
consistent, and any mismatch is a type error.

The two-dimensional convolution layer is a |ParaLens'| with the convolutional
kernel as its parameter:

\begin{code}
convLens  ::  ( ConvSideCheck h kH 1 0 oH
              , ConvSideCheck w kW 1 0 oW
              , T.All KnownNat [inC, outC, h, w, batch, oH, oW]
              , T.KnownDType dt, T.KnownDevice dv )
          =>  ParaLens'
                (Tensor dv dt [outC, inC, kH, kW])
                (Tensor dv dt [batch, inC, h, w])
                (Tensor dv dt [batch, outC, oH, oW])
\end{code}

\noindent The constraints |ConvSideCheck h kH 1 0 oH| and
|ConvSideCheck w kW 1 0 oW| encode the output-dimension equations
\[
  o_H \;=\; h - k_H + 1, \qquad o_W \;=\; w - k_W + 1
\]
for unit stride and zero padding at the type level.  A kernel or input of the
wrong spatial size is a compile-time type error rather than a runtime failure.

The backward pass is split into two computations.  The gradient with respect to
the input is obtained via the transposed convolution---the adjoint of the forward
operator:
\[
  \tfrac{\partial L}{\partial x} \;=\; \mathit{convTranspose2d}(\mathit{kernel},\; \mathit{grad})
\]
% The gradient with respect to the kernel uses the im2col decomposition rather than
% the opaque |convolution_backward_overrideable| internal.  
The function |im2col| unfolds each spatial patch of the input into a column, producing a tensor of shape
|[batch, inC*kH*kW, oH*oW]|.  The kernel gradient is then a batched matrix
multiply, summed over the batch dimension:
\[
  \tfrac{\partial L}{\partial W}
  \;=\; \sum_{\mathit{batch}} \mathit{grad\_flat} \cdot \mathit{x\_col}^\top
  \;\;\in\; \mathbb{R}^{\mathit{outC} \times \mathit{inC} \cdot k_H \cdot k_W}
\]
which is reshaped to |[outC, inC, kH, kW]|.  This makes the gradient derivation
explicit and keeps the backward pass within the verifiable part of the typed API.

\section{Pooling and Shape}

Max pooling carries no parameters and is therefore a plain |Lens'|, with kernel
size, stride, and padding all verified by |ConvSideCheck| at the type level:

\begin{code}
maxPool  ::  ( ConvSideCheck h (Fst kernelSize) (Fst stride) (Fst padding) oH
             , ConvSideCheck w (Snd kernelSize) (Snd stride) (Snd padding) oW )
         =>  Lens' (Tensor device dtype [batch, channels, h, w])
                   (Tensor device dtype [batch, channels, oH, oW])
\end{code}

\noindent The backward pass uses a max-routing mask: the pooled output is expanded
back to the input's spatial resolution via |repeat_interleave|, compared
element-wise with the original input to identify which positions achieved the
maximum, and the incoming gradient flows only through those positions.

The |flatten| lens collapses all non-batch dimensions into a single axis:

\begin{code}
flatten  ::  KnownNat (T.Numel shape)
         =>  Lens' (Tensor dev dt (batch : shape))
                   (Tensor dev dt [batch, T.Numel shape])
\end{code}

\noindent The type-level computation |T.Numel shape| computes the total number of elements in the shape list
at compile time, so the output width is a compile-time constant.  The backward
pass is a plain reshape with no gradient arithmetic.  Like the activation
functions, |maxPool| and |flatten| are plain |Lens'| values and compose
into parametric pipelines with |(.)| rather than |(.#.)|, contributing
no parameter wires to the composed diagram.

\section{Attention}

Self-attention is the most structurally complex layer in this chapter: the
output at every position is a weighted average of all other positions, with
weights that are themselves a differentiable function of the input.
Despite this non-linearity the layer is \emph{stateful}---four square
projection matrices are its learnable parameters---and it fits into the
|ParaLens'| abstraction without modification.  The parameter type groups
the four matrices in a tuple:

\begin{code}
type SelfAttnP dev dt e =
  ( Tensor dev dt [e, e]  -- Wq
  , Tensor dev dt [e, e]  -- Wk
  , Tensor dev dt [e, e]  -- Wv
  , Tensor dev dt [e, e]  -- Wo
  )
\end{code}

\noindent With batch size $b$, sequence length $s$, and embedding dimension $e$
fixed at the type level, the signature of |selfAttention| is:

\begin{code}
selfAttention  ::  ( T.All KnownNat [b, s, e]
                   , T.MatMulDTypeIsValid dev dt
                   , T.BasicArithmeticDTypeIsValid dev dt
                   , T.StandardFloatingPointDTypeValidation dev dt
                   , T.SumDType dt ~ dt, T.SumDTypeIsValid dev dt
                   , KnownNat (b * s)
                   , (b * (s * e)) ~ ((b * s) * e) )
               =>  ParaLens'
                     (SelfAttnP dev dt e)
                     (Tensor dev dt [b, s, e])
                     (Tensor dev dt [b, s, e])
\end{code}

\noindent The equality |(b * (s * e)) ~ ((b * s) * e)| cannot be discharged
automatically by GHC's type-level arithmetic---the solver does not apply
associativity of multiplication without an explicit witness.  It must be
named in the constraint to justify the |reshape| to $[b{\cdot}s,\; e]$
inside the weight-gradient helper.

\paragraph{Forward pass.}
Given input $X \in \mathbb{R}^{b \times s \times e}$, scaled dot-product
attention proceeds in six steps.  The input is first projected into query,
key, and value spaces:
\[
  Q = X W_Q^\top, \qquad K = X W_K^\top, \qquad V = X W_V^\top
  \qquad \in \mathbb{R}^{b \times s \times e}
\]
The attention scores are formed by a scaled inner product, normalised by
softmax, and used to produce a weighted average of the values:
\[
  S = \tfrac{1}{\sqrt{e}}\; Q K^\top
  \;\in \mathbb{R}^{b \times s \times s},
  \qquad
  A = \operatorname{softmax}(S),
  \qquad
  \mathit{out} = A V W_O^\top
\]
All six intermediate values are retained by an internal helper |runFwd|,
which is shared between the getter and the setter so that the backward
pass need not recompute the forward pass:

\begin{code}
    runFwd (wq, wk, wv, wo) x =
      let q       = proj wq x
          k       = proj wk x
          v       = proj wv x
          scores  = T.mulScalar scale $ T.matmul q $ tr k
          weights = T.softmax @2 scores
          attn    = T.matmul weights v
      in (q, k, v, weights, attn, proj wo attn)
\end{code}

\noindent where |proj w x = T.matmul x (T.transpose @0 @1 w)| computes
$x W^\top$, |tr = T.transpose @1 @2| transposes the last two index
positions, and |scale| is the compile-time constant $1/\sqrt{e}$.

\paragraph{Backward pass.}
The setter applies the chain rule in the reverse order of the graph.
At each node the standard matrix-calculus identity for $Y = X A^\top$,
namely $\partial L/\partial X = (\partial L/\partial Y)\, A$ and
$\partial L/\partial A = (\partial L/\partial Y)^\top X$, is applied.

\emph{Output projection.}
Differentiating $\mathit{out} = \mathit{attn}\, W_O^\top$:
\[
  \tfrac{\partial L}{\partial \mathit{attn}}
    = \tfrac{\partial L}{\partial \mathit{out}}\, W_O,
  \qquad
  \tfrac{\partial L}{\partial W_O}
    = \Bigl(\tfrac{\partial L}{\partial \mathit{out}}\Bigr)^\top \mathit{attn}
\]
Both tensors are reshaped to $[b{\cdot}s,\; e]$ before the matrix multiply,
yielding $\partial L/\partial W_O \in \mathbb{R}^{e \times e}$.  The helper
|wGrad| encapsulates this flatten-matmul pattern and is reused identically
for all four projection matrices, playing the same role that
|T.matmul (transp grad) x| plays in |linear|:

\begin{code}
    wGrad x dY =
      T.matmul  (T.transpose @0 @1 (T.reshape @[b * s, e] dY))
                (T.reshape @[b * s, e] x)
\end{code}

\emph{Weighted sum.}
Differentiating $\mathit{attn} = A\, V$:
\[
  \tfrac{\partial L}{\partial A}
    = \tfrac{\partial L}{\partial \mathit{attn}}\, V^\top,
  \qquad
  \tfrac{\partial L}{\partial V}
    = A^\top\, \tfrac{\partial L}{\partial \mathit{attn}}
\]

\emph{Softmax.}
The Jacobian of softmax contracts to a rank-one correction.  For a single
row $y = \operatorname{softmax}(x)$ and incoming gradient $g$:
\[
  \tfrac{\partial L}{\partial x_i}
  = y_i\!\left(g_i - \sum_j y_j\, g_j\right)
\]
The inner dot product is computed with |sumDim @2| along the key axis,
reshaped to $[b, s, 1]$ for broadcasting, and the result is scaled
element-wise by $A$:

\begin{code}
    softmaxBwd w dw = w * T.sub dw dot
      where dot = T.reshape @[b, s, 1] $ T.sumDim @2 (w * dw)
\end{code}

\emph{Scaled dot-product.}
Differentiating $S = \frac{1}{\sqrt{e}}\, Q K^\top$:
\[
  \tfrac{\partial L}{\partial Q}
    = \tfrac{1}{\sqrt{e}}\,\tfrac{\partial L}{\partial S}\, K,
  \qquad
  \tfrac{\partial L}{\partial K}
    = \tfrac{1}{\sqrt{e}}\!\left(\tfrac{\partial L}{\partial S}\right)^\top Q
\]

\emph{Input projections.}
|wGrad| is applied once per projection to obtain
$\partial L/\partial W_Q$, $\partial L/\partial W_K$, and
$\partial L/\partial W_V$.  The gradient with respect to the input $X$
accumulates the three back-projections:
\[
  \tfrac{\partial L}{\partial X}
  = \tfrac{\partial L}{\partial Q}\, W_Q
  + \tfrac{\partial L}{\partial K}\, W_K
  + \tfrac{\partial L}{\partial V}\, W_V
\]
The complete setter, with |mm = T.matmul|, |tr = T.transpose @1 @2|,
and |scale'| denoting multiplication by $1/\sqrt{e}$, is:

\begin{code}
    rev (p@(wq,wk,wv,wo), x) dOut = ((dWq, dWk, dWv, dWo), dX)
      where
        (q, k, v, weights, attn, _) = runFwd p x
        dAttn    = T.matmul dOut wo
        dWo      = wGrad attn dOut
        dWeights = mm dAttn    (tr v)
        dV       = mm (tr weights) dAttn
        dScores  = softmaxBwd weights dWeights
        dQ       = scale' $ mm dScores    k
        dK       = scale' $ mm (tr dScores) q
        dWq      = wGrad x dQ
        dWk      = wGrad x dK
        dWv      = wGrad x dV
        dX       = T.matmul dQ wq + T.matmul dK wk + T.matmul dV wv
\end{code}

\paragraph{Multi-head attention.}
|multiHeadSelfAttention| partitions the embedding dimension across $h$
independent attention heads, each of width $\mathit{hd} = e / h$, enforced
at the type level by |e ~ h * hd|.  The parameter type is unchanged---the
weight matrices remain $[e, e]$ and operate on the full embedding---but two
reshape helpers split and merge the head dimension around the attention
computation:

\begin{code}
    splitHeads  = T.transpose @1 @2 . T.reshape @[b, s, h, hd]
    mergeHeads  = T.reshape @[b, s, e] . T.transpose @1 @2
\end{code}

\noindent |splitHeads| maps $[b, s, e] \to [b, s, h, \mathit{hd}] \to
[b, h, s, \mathit{hd}]$: the |reshape| partitions each embedding vector
into $h$ contiguous slices of width $\mathit{hd}$, and the |transpose|
brings the head axis adjacent to the batch axis so that subsequent batched
matrix multiplies treat $b \times h$ as a single leading batch dimension.
The constraint |Numel '[b, s, e] ~ Numel '[b, s, h, hd]| is the type-level
proof that the reshape preserves the total number of elements.

Two details differ from the single-head case.  First, the scale factor is
$1/\sqrt{\mathit{hd}}$ rather than $1/\sqrt{e}$: within each head the
key--query dot products grow with the per-head width $\mathit{hd}$, not
with the full embedding $e$.  Second, softmax is applied along dimension~3
(the key-position axis in the layout $[b, h, s, s]$) rather than
dimension~2, and the softmax backward uses |sumDim @3| and a broadcast
reshape to $[b, h, s, 1]$ accordingly.

The backward pass is structurally identical to the single-head case.
Gradients for $Q$, $K$, and $V$ are computed in the $[b, h, s, \mathit{hd}]$
layout and collapsed to $[b, s, e]$ via |mergeHeads| before being passed
to |wGrad|.  Because the projection matrices remain $[e, e]$ regardless
of $h$, |wGrad| is shared unchanged between the two implementations.

Like every other layer in this chapter, both attention variants are
|ParaLens'| values and compose into larger architectures via |(.#.)|.
The 4-tuple parameter type is absorbed automatically into the product type
of the enclosing model by the composition rule, with no boilerplate
required.
