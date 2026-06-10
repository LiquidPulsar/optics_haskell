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
module CaseStudies where
\end{code}
%endif

\chapter{Case Studies}
\label{chap:casestudies}

The preceding chapters described the framework's abstractions in
isolation.  This chapter grounds them in three concrete applications.
The first, Iris flower classification, is a standard multi-layer
perceptron benchmark small enough to inspect the output of GHC's
optimiser directly.  The second, MNIST digit recognition, shows the
same compositional style scaling to a convolutional architecture.
The third revisits MNIST with a residual architecture, demonstrating
that |skipPara| composes into larger models without modification and
that its parameter type is inferred automatically by the type system.
Together they demonstrate that the lens abstraction adds zero runtime
cost and that the type system catches architectural mistakes at compile
time rather than at runtime.

\section{Iris Flower Classification}
\label{sec:iris}

The Iris dataset comprises 150 specimens of three species of iris,
each described by four real-valued morphological measurements (sepal
and petal length and width).  The classification task is a standard
MLP benchmark and is used here specifically because the network is
small enough to read GHC Core output in full and compare it line for
line against a hand-written implementation.

\subsection*{Model Definition}

The architecture is $n$ hidden fully-connected layers of width~4,
followed by a single output layer that maps 4 features to 3 class
logits.  Three type aliases encode this at the type level:

\begin{code}
type InnerLayer dv dt    =  MMP dv dt 4 4
type IParams    n dv dt  =  (StackedN n (InnerLayer dv dt), MMP dv dt 3 4)
\end{code}

\noindent |MMP dv dt out in = (T.Tensor dv dt [out, in], T.Tensor dv dt [out])|
is a weight matrix paired with a bias vector (Chapter~\ref{chap:layers}).
|InnerLayer| is a $4\!\to\!4$ affine map.  |IParams n dv dt| is the
product of $n$ inner layers followed by one output layer; for $n = 2$
it expands to
|((MMP dv dt 4 4, MMP dv dt 4 4), MMP dv dt 3 4)|---a fully concrete
nested tuple whose structure mirrors the layer order.

The |ParaLens'| model is:

\begin{code}
nMuls  ::  (CanStack n, SaneDT dv dt) =>
           ParaLens'  (StackedN n (MMP dv dt m m))
                      (T.Tensor dv dt [b, m])
                      (T.Tensor dv dt [b, m])
nMuls  =   stackN @n (matMulLens . sigmoid)

irisModel  ::  (CanStack n, SaneDT dv dt) =>
               ParaLens'  (Inp (T.Tensor dv dt [b, 4]), IParams n dv dt)
                          ()
                          (Out (T.Tensor dv dt [b, 3]))
irisModel  =   argToPara .#. nMuls @n .#. matMulLens . sigmoid
\end{code}

\noindent |argToPara| (Section~\ref{sec:repara}) converts the raw input
tensor into a |ParaLens'| with a unit parameter, making it composable
with |(.#.)|.  Each |(.#.)| join adds one more layer, automatically
extending the parameter type.  The final |matMulLens . sigmoid| is the
output layer---relu is parameter-free so it composes with plain |(.)|
rather than with |(.#.)|, leaving the output layer's parameter type
clean.

The hand-rolled forward pass for $n = 1$ is:

\begin{code}
handRolledTest (i, (mb, mb'))  =  layer mb' . layer mb $ i
  where
    layer (m, b)  =  T.sigmoid . T.add b . (`T.matmul` transp m)
\end{code}

\noindent The two definitions produce identical GHC Core, as shown
in the next subsection.

\subsection*{Zero-Overhead Abstraction}
\label{sec:zero-overhead}

The elimination happens in two stages.  First, the van Laarhoven
representation
\[
  |Lens s t a b| \;=\; \forall f.\; \mathit{Functor}\;f \Rightarrow (a \to f\,b) \to s \to f\,t
\]
is specialised at the call site.  The forward pass (|runFullModel|)
instantiates $f = \mathtt{Const}\;b$, for which
|fmap = flip const|, collapsing every lens in the chain
to its getter.  The backward pass instantiates $f = \mathtt{Identity}$,
for which |fmap = coerce|, collapsing to the setter--getter
chain.  In both cases the |forall f| disappears and all intermediate
|fmap| calls and |(,)| wrappers reduce to their concrete definitions.
Second, the |{-# INLINE #-}| pragmas on |(.#.)|, |repara|, |stackN|,
and their auxiliaries (|swapFst|, |rotate|, |leftLens|, |rightLens|)
allow GHC to discharge the remaining composition layers, leaving only
the raw tensor operations.

Compiling both |test = runFullModel (irisModel @1)| and
|handRolledTest| with \texttt{-O2} and \texttt{-ddump-simpl} produces
worker functions \texttt{\$wtest} and \texttt{\$whandRolledTest} that
are structurally identical.

Each worker accepts nine unboxed arguments---a \texttt{ForeignPtr} pair
for the input tensor and two pointer pairs per weight/bias group---and
performs the same eight operations in the same order:

\begin{verbatim}
  $s$wtranspose  w1_ptr              --  W1^T
  $wmatmul_tt    x_ptr    W1^T       --  X * W1^T
  $wadd          b1_ptr   (X*W1^T)   --  + b1
  $wsigmoid_t    (X*W1^T + b1)       --  sigma_1
  $s$wtranspose  w2_ptr              --  W2^T
  $wmatmul_tt    h1_ptr   W2^T       --  h1 * W2^T
  $wadd          b2_ptr   (h1*W2^T)  --  + b2
  $wsigmoid      (h1*W2^T + b2)      --  sigma_2
\end{verbatim}

\noindent The wrapper functions also match: both \texttt{test1} and
\texttt{handRolledTest} perform twelve nested \texttt{case} analyses to
unpack the input tuple into nine raw pointer arguments, then tail-call
their respective workers.  The only differences in the entire output
are internal variable names and coercion-tag numbers
(\texttt{Co:42} vs.\ \texttt{Co:960} for the final cast), which are
GHC-internal type-representation artefacts with no runtime cost.

The lens machinery---|Lens (p,a)|, |(.#.)|, |alongside|, |swapFst|,
|rotate|, the |(,)| tuple constructors---is entirely absent from the
optimised output.  The user writes four lines of lens composition; GHC
emits the same sequence of LibTorch FFI calls as the hand-written
version.

\subsection*{Type-Level Layer Depth}
\label{sec:stack}

The depth $n$ is a compile-time natural number, not a runtime value.
\texttt{Stack.hs} converts it to a Peano numeral and uses typeclass
induction to build both the parameter type and the |(.#.)| composition
tree simultaneously:

\begin{code}
data PeanoNat = One | Succ PeanoNat

type family Stacked (n :: PeanoNat) (m :: Type) where
    Stacked One      m  =  m
    Stacked (Succ n) m  =  (m, Stacked n m)

class StackN (n :: PeanoNat) where
    stack  ::  ParaLens' p a a -> ParaLens' (Stacked n p) a a

instance StackN One where
    stack l  =  l

instance StackN n => StackN (Succ n) where
    stack l  =  l .#. stack @n l
\end{code}

\noindent The public API bridges GHC's built-in |Nat| literals:

\begin{code}
type StackedN  (n :: Nat) m  =  Stacked (ToPeano n) m
type CanStack  n             =  StackN  (ToPeano n)

stackN  ::  CanStack n => ParaLens' p a a -> ParaLens' (StackedN n p) a a
stackN  =   stack @(ToPeano n)
\end{code}

\noindent For $n = 3$, |StackedN 3 (MMP dv dt 4 4)| resolves at
compile time to |(MMP dv dt 4 4, (MMP dv dt 4 4, MMP dv dt 4 4))|.
Crucially, GHC always sees a concrete product type, not a list or a
vector.  This is precisely what enables the zero-overhead property
from the previous subsection: because the parameter type is a fully
known nested tuple at each $n$, the inliner can fully specialise and
unbox the entire parameter structure, producing the same Core as a
hand-written $n$-layer network at any fixed depth.

Adding or removing a hidden layer is a one-character change to the
type application |@n|; no other code changes.

The same typeclass induction extends to initialisation.  A |RandInit|
instance on pairs lifts random sampling to any product type
automatically:

\begin{code}
instance (RandInit l, RandInit r) => RandInit (l, r) where
    randInit  =  liftA2 (,) randInit randInit
\end{code}

\noindent |irisInitParams @n| therefore generates a freshly randomised
|IParams n dv dt| of the correct depth with no boilerplate.

\subsection*{Devices, Dtypes, and Static Shapes}

The tensor type |T.Tensor dv dt shape| carries three compile-time
parameters: the device |dv|, the element dtype |dt|, and the shape
|shape| (a type-level list of dimension sizes).  The |SaneDT|
constraint alias bundles the training requirements:

\begin{code}
type SaneDT dv dt  =
  (  T.KnownDType dt
  ,  T.StandardFloatingPointDTypeValidation dv dt
  ,  CanMMLens dv dt  )
\end{code}

\noindent |T.StandardFloatingPointDTypeValidation dv dt| is unsatisfied
by integer or boolean dtypes, so any attempt to train with a
non-differentiable dtype is rejected by the type checker before a
single line of training code runs.  Switching between precisions or
between CPU and GPU requires only a change to the type application at
the call site:

\begin{code}
irisGetEpoch @1 @(T.CPU,  0) @T.Double  -- CPU, double precision
irisGetEpoch @1 @(T.CUDA, 0) @T.Float   -- GPU, single precision
\end{code}

\noindent The model definition, training loop, and loss function are
entirely unchanged.  Shape mismatches---for example, feeding a
$[\mathit{batch},10]$ tensor to a layer expecting $[\mathit{batch},4]$---
are type errors caught by the shape in |T.Tensor dv dt [b, 4]|.  No
runtime assertions or dynamic shape checks are needed anywhere in the
framework.

The small size of the Iris model means that the performance differences
between Float and Double are neglibible, see Table~\ref{tab:mnist-perf}
below for a more realistic benchmark on the larger MNIST dataset.

\section{MNIST Digit Recognition}
\label{sec:mnist}

MNIST is a dataset of 70{,}000 greyscale images of handwritten digits
(0--9) at $28\times28$ pixels, split into 60{,}000 training and
10{,}000 test examples.  The model used here is a small convolutional
network followed by a dense output layer:

\begin{verbatim}
[batch,  1, 28, 28]
  conv1 (3x3, 3 filters)  -> relu -> maxpool(2x2)  -> [batch,  3, 13, 13]
  conv2 (4x4, 5 filters)  -> relu -> maxpool(2x2)  -> [batch,  5,  5,  5]
  flatten                                           -> [batch, 125]
  dense (125 -> 10)       -> sigmoid                -> [batch,  10]
\end{verbatim}

\noindent The 1{,}527-parameter model (conv1: 27, conv2: 240,
dense: 1{,}260) is intentionally modest; the purpose is to
demonstrate the framework's compositional style on a multi-stage
architecture, not to achieve state-of-the-art accuracy.

\subsection*{Performance}

\begin{table}[h]
\centering
\begin{tabular}{lccccc}
\toprule
Precision       & Mean (ms) & Min (ms) & Max (ms) & Std dev (ms) & $R^2$ \\
\midrule
|T.Float|  (32-bit) & 251 & 232 & 260 & 16 & 0.993 \\
|T.Double| (64-bit) & 289 & 284 & 292 &  5 & 1.000 \\
\bottomrule
\end{tabular}
\caption{MNIST CNN training: wall-clock time per epoch by floating-point
  precision, CPU, batch size 32, 6{,}000 training examples.
  Measured with Criterion after a full-epoch warmup pass for each dtype;
  min/max are the 95\% confidence bounds on the mean.
  Float is 15\% faster, consistent with wider SIMD throughput for 32-bit arithmetic.}
\label{tab:mnist-perf}
\end{table}

\subsection*{Parameter Types}

Four type aliases encode the architecture at the type level:

\begin{code}
type Conv1K dev dt  =  T.Tensor dev dt [3, 1, 3, 3]
type Conv2K dev dt  =  T.Tensor dev dt [5, 3, 4, 4]
type DenseP dev dt  =  MMP dev dt 10 125
type MnistP dev dt  =  (Conv1K dev dt, (Conv2K dev dt, DenseP dev dt))
\end{code}

\noindent |MnistP| is the product of all learnable parameters in layer
order.  A kernel of the wrong shape---say, |[3, 1, 4, 4]| instead of
|[3, 1, 3, 3]| for |Conv1K|---is a compile-time type error.  The
product structure is assembled automatically by |(.#.)|: each
composition step extends the parameter tuple by one more layer's worth
of weights, so |MnistP| is inferred rather than written by hand.

The |SaneMnist| constraint alias bundles every capability required for
training, including dtype validity for mean reduction (used by the loss)
and comparison operations (used by the accuracy metric):

\begin{code}
type SaneMnist dev dt  =
  (  CanMMLens dev dt
  ,  T.StandardFloatingPointDTypeValidation dev dt
  ,  T.MeanDTypeValidation dev dt
  ,  T.KnownDevice dev
  ,  T.KnownDType dt  )
\end{code}

\noindent Any device/dtype combination that fails to satisfy the full
set---for example, a device with no mean-reduction support---is
rejected at compile time.

\subsection*{Model Composition}

\begin{code}
mnistModel  =   argToPara
  .#.  withGradDesc convLens . relu . maxPool @(2, 2) @(2, 2) @(0, 0)
  .#.  withGradDesc convLens . relu . maxPool @(2, 2) @(2, 2) @(0, 0)
  .#.  rightLens (flatten @b @[5, 5, 5]) . matMulLens
\end{code}

\noindent Each |(.#.)| arm contributes one block to both the forward
pass and the combined parameter type.  Within each convolutional arm,
|relu| and |maxPool @...| are parameter-free lenses composed with
plain |(.)| rather than |(.#.)|, so they thread data through without
extending the parameter type.  The type applications to |maxPool|---
kernel size, stride, and padding---are static; the output shape after
each pooling step is checked at compile time.  |withGradDesc convLens|
(Section~\ref{sec:gradupdate}) wraps each convolutional layer with the
CRDC natural addition, so the gradient update $\theta \leftarrow \theta
+ \partial\theta$ is baked into the layer itself.

The final arm uses |rightLens| to pre-process the data component of
the |(param,\,data)| pair before the dense layer:

\[
  (\mathit{MMP},\; [\mathit{batch},5,5,5])
  \;\xrightarrow{\mathtt{rightLens}\;(\mathtt{flatten})}\;
  (\mathit{MMP},\; [\mathit{batch},125])
  \;\xrightarrow{\mathtt{matMulLens}}\;
  [\mathit{batch},10].
\]

\noindent |flatten| is a pure |Iso'| with no learnable parameters and
no effect on the gradient path through |MMP|.  Composing it via
|rightLens| (which lifts a lens on the second component of a pair)
keeps the reshape and the matmul as a single |ParaLens'| unit, avoiding
any intermediate allocation.

\subsection*{He Initialisation}

\begin{code}
mnistInitParams :: IO (MnistP dev dt)
mnistInitParams = do
  let sc x  =  T.mulScalar (x :: Float)
  c1  <-  sc (sqrt (2 / 9))    <$>  T.randn
  c2  <-  sc (sqrt (2 / 48))   <$>  T.randn
  w   <-  sc (sqrt (2 / 125))  <$>  T.randn
  let b  =  T.zeros
  pure (c1, (c2, (w, b)))
\end{code}

\noindent He initialisation~\citep{he2015delving} sets each layer's
weight standard deviation to $\sqrt{2/\text{fan\_in}}$, placing initial
activations in the near-linear region of the sigmoid and preventing
vanishing gradients in early training.  The fan-in values are
$1\times3\times3 = 9$ for |Conv1K|, $3\times4\times4 = 48$ for
|Conv2K|, and $125$ for the dense weight matrix.  The bias is
initialised to zero.

\subsection*{Loss and Training}

The loss and training models follow the same pattern as Iris:

\begin{code}
mnistModelLoss  =  mnistModel .#. softMaxCELoss
mnistModel'     =  mnistModel .#. softMaxCELoss . lrSmooth 1e-4
\end{code}

\noindent |softMaxCELoss| (Section~\ref{sec:celoss}) fuses the softmax
and log operations for numerical stability.  |lrSmooth 1e-4| caps the
output wire to the unit type, seeding backpropagation with
$-10^{-4}$.

One training epoch is a strict left fold over all batches:

\begin{code}
mnistEpoch trainT params  =  trainMany mnistModel' params trainT
\end{code}

\noindent Inference extracts predictions by taking the argmax along the
class axis:

\begin{code}
mnistPredict p x  =
  U.asValue . T.toDynamic $
    T.argmax @1 @T.DropDim (runFullModel mnistModel (x, p))
\end{code}

\noindent The |@1| selects the class dimension and |@T.DropDim| removes
it, returning a list of integer digit labels.  The type of
|runFullModel mnistModel (x, p)| is |T.Tensor dev dt [b, 10]|, so the
argmax and the label extraction are statically typed throughout.

\section{Residual MNIST}
\label{sec:resmnist}

The MNIST task is revisited with a residual architecture to show that
|skipPara| (Section~\ref{sec:skipconnections}) integrates into a
larger pipeline without modification.  The model replaces the
convolutional stack with a sequence of dense residual blocks, keeping
the setting comparable to the plain MNIST model.

\subsection*{Parameter Types}

Each residual block wraps two affine layers of the same width,
so its parameter type is a pair of weight-bias pairs:

\begin{code}
type Hidden    = 128
type NumBlocks = 3

type ResBlockP dev dt  =  (MMP dev dt Hidden Hidden, MMP dev dt Hidden Hidden)

type ResMnistP dev dt  =
  (  MMP dev dt Hidden 784
  ,  (StackedN NumBlocks (ResBlockP dev dt), MMP dev dt 10 Hidden)  )
\end{code}

\noindent |ResMnistP| is the product of an input projection, three
stacked block parameter pairs, and an output layer, assembled
automatically by the |(.#.)| composition rule.

\subsection*{Model Composition}

A single residual block is one application of |skipPara|:

\begin{code}
resBlock  ::  SaneRes b dev dt
          =>  ParaLens'  (ResBlockP dev dt)
                         (Tensor dev dt [b, Hidden])
                         (Tensor dev dt [b, Hidden])
resBlock  =   skipPara (matMulLens . relu .#. matMulLens . relu)
\end{code}

\noindent The parameter type |(ResBlockP dev dt)| is inherited from the
inner pipeline; |skipPara| contributes no additional parameters.
The full model stacks |NumBlocks| such blocks between a flattening
input projection and a sigmoid output layer:

\begin{code}
resMnistModel  =   argToPara
  .#.  rightLens (flatten @b @[1, 28, 28]) . matMulLens . relu
  .#.  stackN @NumBlocks resBlock
  .#.  matMulLens . sigmoid
\end{code}

\noindent The pipeline is a drop-in replacement for |mnistModel|: the
same training loop, loss combinator, and inference infrastructure apply
unchanged.

\subsection*{He Initialisation}

Weights are initialised with standard deviation $\sqrt{2/\text{fan\_in}}$.
The fan-in for the input projection is 784; for every layer inside a
residual block it is |Hidden = 128|.

\begin{code}
resMnistInitParams = do
  let sc x  =  T.mulScalar (x :: Float)
  wIn   <-  sc (sqrt (2 / 784))                     <$>  T.randn
  -- StackedN 3 (ResBlockP) = (block1, (block2, block3))
  w1a   <-  sc (sqrt (2 / natValF @Hidden))         <$>  T.randn
  -- ... (w1b, w2a, w2b, w3a, w3b initialised identically)
  wOut  <-  sc (sqrt (2 / natValF @Hidden))         <$>  T.randn
  pure (...)
\end{code}

\subsection*{Longer-Range and Projection Skips}

The architecture above chains three blocks with |stackN|, each
carrying an independent |skipPara| connection spanning its own two
layers---the standard residual block structure of He et al.\
\citep{he2015delving}.  The blocks are composed sequentially; there
are no cross-block skip connections, matching the original ResNet
design.

A limitation of |skipPara| as defined is that it requires the
sub-network to be \emph{type-preserving}: |f :: ParaLens' p a a|.
Real ResNet stages change both spatial resolution and channel count
across some transitions, handled in the original paper by a
\emph{projection shortcut} $y = f(x) + P(x)$ where $P$ is a $1\times1$
convolution that matches dimensions.  The natural generalisation in
this framework is a combinator that runs $f$ and $P$ in parallel and
sums:

\begin{code}
projSkipPara  ::  (Num b, Num b')
              =>  ParaLens p p' a a' b b'
              ->  ParaLens q q' a a' b b'
              ->  ParaLens (p, q) (p', q') a a' b b'
projSkipPara f proj  =  alongside (leftLens f) (leftLens proj)
                          . from (toPara splitIso)
\end{code}

\noindent |projSkipPara f id| recovers |splitPara f| (dual fan-out of
the output); |projSkipPara f proj| is the full projection shortcut.
Same-type skips---the common case---do not require the projection
parameter, so |skipPara| remains the right interface for
dimensionality-preserving blocks.
