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

\chapter{Case Studies and Evaluation}
\label{chap:casestudies}

The preceding chapters described the framework's abstractions in
isolation.  This chapter grounds them in four concrete applications.
The first, Iris flower classification, is a standard multi-layer
perceptron benchmark small enough to inspect the output of GHC's
optimiser directly.  The second, MNIST digit recognition, shows the
same compositional style scaling to a convolutional architecture.
The third demonstrates that the framework is not limited to
discriminative tasks: an MNIST autoencoder is expressed as two
ordinary |ParaLens'| pipelines composed end-to-end, with the
bottleneck constraint enforced at the type level.  The fourth revisits
MNIST with a residual architecture, demonstrating that |skipPara|
composes into larger models without modification and that its parameter
type is inferred automatically by the type system.
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

\noindent For $n = 3$, |StackedN 3 (MMP ..)| resolves at
compile time to |(MMP .., (MMP .., MMP ..))|.
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
between Float and Double are negligible; Table~\ref{tab:mnist-perf}
in Section~\ref{sec:mnist} gives a more realistic benchmark on the
larger MNIST dataset.

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
  dense (125 -> 10)                                 -> [batch,  10]
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
weight standard deviation to $\sqrt{2/\text{fan\_in}}$, keeping the
variance of activations constant across ReLU layers and preventing
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

\subsection*{Training Results}

The model was trained for 400 epochs on a 6{,}000-example subset of
the MNIST training set (10\%)---chosen to keep each epoch fast enough
to run on a CPU-only machine---with batch size~32 and learning rate
$10^{-4}$.  Test accuracy was evaluated on the full 10{,}000
example test set after each epoch.  The model reaches approximately
\textbf{90\% test accuracy} around epoch~300, peaking at 90.4\% at
epoch~299 and plateauing thereafter.  Given that the model has only
1{,}527 parameters and is trained on one-tenth of the available data,
the result confirms that the framework's compositional forward and
backward passes are numerically correct end-to-end and that the lens
abstraction imposes no obstacle to learning.

\section{MNIST Autoencoder}
\label{sec:autoencoder}

Autoencoders demonstrate that the framework is not limited to
discriminative tasks.  An autoencoder learns to compress its input
into a low-dimensional latent representation and then reconstruct the
original from that representation, trained by minimising reconstruction
error rather than classification loss.  In the parametric lens
framework, encoder and decoder are each ordinary |ParaLens'|
pipelines; composing them end-to-end with |(.#.)| produces the full
model with no special casing.

\subsection*{Architecture}

The model maps flattened MNIST images ($28\times28 = 784$ pixels) to a
32-dimensional latent space and back:

\begin{verbatim}
  encoder: [b, 784] -> Dense(784->128) -> ReLU -> Dense(128->32)
  decoder: [b,  32] -> Dense(32->128)  -> ReLU -> Dense(128->784) -> Sigmoid
\end{verbatim}

\noindent The bottleneck---latent dimension 32 strictly smaller than
input dimension 784---is enforced at the type level: the encoder output
type |T.Tensor dv dt [b, LatentDim]| must unify with the decoder input
type, so a dimension mismatch is a compile-time error rather than a
silent shape broadcast.

\subsection*{Parameter Types}

Three type aliases capture the parameter structure:

\begin{code}
type EncoderP dev dt  =  (MMP dev dt HiddenDim InputDim,
                          MMP dev dt LatentDim HiddenDim)
type DecoderP dev dt  =  (MMP dev dt HiddenDim LatentDim,
                          MMP dev dt InputDim  HiddenDim)
type AEP      dev dt  =  (EncoderP dev dt, DecoderP dev dt)
\end{code}

\noindent where |InputDim = 784|, |HiddenDim = 128|, and
|LatentDim = 32|.  |AEP dev dt| is a nested product of four
weight-bias pairs.  As with the discriminative models, this type is not
written by hand: it is assembled automatically by |(.#.)| from the
encoder and decoder parameter types.

\subsection*{Model Composition}

Encoder and decoder are each plain |ParaLens'| values:

\begin{code}
encoderCore  ::  (SaneAE dev dt, KnownNat b)
             =>  ParaLens'  (EncoderP dev dt)
                            (T.Tensor dev dt [b, InputDim])
                            (T.Tensor dev dt [b, LatentDim])
encoderCore  =   matMulLens . relu .#. matMulLens

decoderCore  ::  (SaneAE dev dt, KnownNat b)
             =>  ParaLens'  (DecoderP dev dt)
                            (T.Tensor dev dt [b, LatentDim])
                            (T.Tensor dev dt [b, InputDim])
decoderCore  =   matMulLens . relu .#. matMulLens . sigmoid
\end{code}

\noindent The full autoencoder is a single further composition:

\begin{code}
autoencoderModel  =  argToPara .#. encoderCore .#. decoderCore
\end{code}

\noindent Three |(.#.)| calls assemble the complete pipeline.  The combined
parameter type\linebreak|(T.Tensor dev dt [b, InputDim], AEP dev dt)|
is inferred without annotation.  The composition rule pairs each
sub-lens's parameter wire into the product type automatically, in
exactly the same way as the discriminative models of the preceding
sections.

\subsection*{Loss and Training}

Reconstruction quality is measured by mean-squared error.  The
trainable model is:

\begin{code}
autoencoderModel'  =  autoencoderModel .#. lossSmooth . lrSmooth 1e-3
\end{code}

\noindent |lossSmooth| (Section~\ref{sec:mse}) replaces the
cross-entropy loss of earlier models; no other part of the training
infrastructure changes.  Each training step presents a batch |x| of
flattened images paired with itself as the reconstruction target:

\begin{code}
aeEpoch trainT  =  flip (trainMany (autoencoderModel' @BatchSize)) trainT
\end{code}

\noindent where |trainT| is a list of |(x, x)| pairs.  The parameter
update, gradient flow, and epoch structure are handled identically to
the discriminative case.  The only difference visible at the call site
is the loss: |lossSmooth| computes $\tfrac{1}{N}\sum_i(p_i - t_i)^2$
rather than cross-entropy, and the gradient seed flows backward through
the decoder and into the encoder without any additional wiring.

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
input projection and a linear output layer:

\begin{code}
resMnistModel  =   argToPara
  .#.  rightLens (flatten @b @[1, 28, 28]) . matMulLens . relu
  .#.  stackN @NumBlocks resBlock
  .#.  matMulLens
\end{code}

\noindent The pipeline is a drop-in replacement for |mnistModel|: the
same training loop, loss combinator, and inference infrastructure apply
unchanged.

\subsection*{He Initialisation}

Weights use the same $\sqrt{2/\text{fan\_in}}$ rule as the MNIST model
above, with fan-in 784 for the input projection and |Hidden = 128| for
every layer inside a residual block.

\subsection*{Longer-Range Skips}

The architecture above chains three blocks with |stackN|, each
carrying an independent |skipPara| connection spanning its own two
layers---the standard residual block structure of He et al.\
\citep{he2015delving}.  The blocks are composed sequentially with no
cross-block skip connections, matching the original ResNet design.
For stages that change spatial resolution or channel count, the
framework provides |projSkipPara| (Section~\ref{sec:projskip}), which
runs a learned projection shortcut in parallel with |f| using the same
|splitIso| fan-out and sum structure.

\section{Performance Evaluation}
\label{sec:perf}

Three implementations of the Iris two-layer MLP at depth $n = 1$ are
compared using Criterion~\cite{criterion} on a CPU-only machine
(GHC~9.8.4, LibTorch~2.9.1):

\begin{description}
  \item[\texttt{typed-hasktorch}] The |ParaLens| library with
    |Torch.Typed| tensors; forward and backward passes are pure Haskell
    lens composition, calling LibTorch only for primitive tensor operations.
  \item[\texttt{hmatrix}] A second lens-based implementation wrapping
    LAPACK and BLAS via the HMatrix library~\cite{hmatrix}; the backward
    pass is also pure Haskell.
  \item[\texttt{dynamic-hasktorch}] A baseline delegating both passes to
    PyTorch autograd via |runStep|, |flattenParameters|, and
    \texttt{.backward()}.
\end{description}

\texttt{raw\_fwd} measures a single forward pass over one batch of eight
samples; \texttt{epoch} measures one full training epoch---a strict left
fold over all 18 mini-batches---including the backward pass and parameter
update for every batch.

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
Benchmark & Implementation & Mean time & Rel.\ to typed \\
\midrule
\texttt{raw\_fwd} & \texttt{typed-hasktorch}          & $6.9\;\mu s$     & $1.00\times$ \\
                  & \texttt{typed-hasktorch-handroll} & $6.8\;\mu s$     & $0.99\times$ \\
\midrule
\texttt{epoch}    & \texttt{hmatrix}                  & $390\;\mu s$     & $0.57\times$ \\
                  & \texttt{typed-hasktorch}          & $689\;\mu s$     & $1.00\times$ \\
                  & \texttt{dynamic-hasktorch}        & $3{,}631\;\mu s$ & $5.27\times$ \\
\bottomrule
\end{tabular}
\caption{Criterion benchmark results for the Iris MLP ($n=1$, batch size~8,
  18 batches per epoch, CPU).  All times are wall-clock means; standard
  deviations were below 4\% for all measurements.}
\label{tab:benchmarks}
\end{table}

\noindent The \texttt{raw\_fwd} row confirms the GHC Core result of
Section~\ref{sec:zero-overhead}: the typed and hand-rolled implementations
are statistically indistinguishable.

\begin{table}[h]
\centering
\begin{tabular}{llll}
\toprule
Variant & Mean time & Diff.\ from baseline & Overhead identified \\
\midrule
\texttt{fold-foldM}        & $3{,}632\;\mu s$ & ---                  & baseline \\
\texttt{fold-foldl'}       & $3{,}649\;\mu s$ & $+17\;\mu s$ (noise) & fold structure: $\approx 0$ \\
\texttt{forward-only}      & $251\;\mu s$     & $-3{,}381\;\mu s$    & \texttt{.backward()} + update \\
\texttt{flattenParams-x18} & $<1\;\mu s$      & ---                  & Generic traversal: $<0.03\;\mu s$ \\
\midrule
\multicolumn{2}{l}{Typed forward $\times 18$}                   & $18 \times 6.9 = 124\;\mu s$ & tensor arithmetic \\
\multicolumn{2}{l}{\texttt{forward-only} $-$ typed$\times$18}  & $251 - 124 = 127\;\mu s$     & autograd graph build \\
\bottomrule
\end{tabular}
\caption{Overhead isolation for one dynamic epoch.  The
  \texttt{forward-only} variant runs the full forward pass (including
  autograd graph construction) but never calls
  \texttt{\char46 backward()}.  Std devs: fold variants $<1\%$,
  \texttt{forward-only} 4\%.}
\label{tab:overhead}
\end{table}

\subsection*{Typed lens vs.\ dynamic autograd}

The |ParaLens| training loop (689~$\mu$s) is \textbf{5.3$\times$ faster}
than the dynamic autograd baseline (3{,}631~$\mu$s).
Table~\ref{tab:overhead} isolates each proposed source of overhead.

\paragraph{PyTorch backward pass: 93\% of the cost.}
The \texttt{forward-only} variant runs the full forward pass for every
mini-batch---including autograd graph construction, since the model
parameters carry |requires_grad=True|---but never calls
|.backward()|.  It completes in $251\;\mu$s.  The full epoch
(3{,}631~$\mu$s) costs $3{,}381\;\mu$s more: \textbf{93\%} of the
total epoch time is spent inside PyTorch's C++ backward pass.  This is
the dominant cost of delegating differentiation to an external autograd
engine.  The |ParaLens| backward is a chain of Haskell closures that
GHC inlines and optimises at compile time; at runtime it reduces to the
same eight LibTorch FFI calls as the forward pass.

\paragraph{Autograd graph construction: 3.5\%.}
Subtracting the typed forward cost ($18 \times 6.9 = 124\;\mu$s) from
the forward-only time ($251\;\mu$s) leaves $127\;\mu$s attributable to
PyTorch's tape construction: allocating |Node| objects, storing input
pointers, and registering backward functions for each of the eight
tensor operations in the two-layer network.  Real but modest---under a
quarter of the cost of the backward pass itself.

\paragraph{Fold structure and Generic traversal: negligible.}
Replacing |foldM| with an explicit |foldl'| chain changes the epoch
time by $17\;\mu$s---within the noise floor.  GHC compiles both to the
same loop at \texttt{-O2}.  Likewise, 18 calls to
|flattenParameters|---the |Generic| traversal collecting the four model
tensors into a list---complete in under $1\;\mu$s total ($<0.03\%$ of
the epoch).

\subsection*{Typed lens vs.\ HMatrix}

The HMatrix baseline (390~$\mu$s) is \textbf{1.8$\times$ faster} than
the typed lens implementation (689~$\mu$s) on Iris.  HMatrix calls
LAPACK and BLAS directly~\cite{hmatrix} and incurs lower per-call
overhead than LibTorch for very small matrices: every LibTorch tensor
operation passes through ATen's multi-device operator
dispatch~\cite{paszke2019pytorch} before reaching the underlying BLAS
kernel, a fixed cost that dominates for $4\times4$ matrices and is
well-studied in the literature~\cite{frison2020blasfeo}.

Both implementations perform the backward pass in pure Haskell; the
per-call FFI cost is the sole driver of the gap.  For larger matrices
or batch sizes the typed lens implementation is expected to match
HMatrix and exceed it via LibTorch's vectorised kernels.

\section{Gradient Correctness Tests}
\label{sec:grad-tests}

The framework includes a test suite that cross-validates every primitive
layer's forward and backward pass against PyTorch autograd, providing
numerical evidence that the hand-derived gradients are correct.

\subsection*{Method}

Each test constructs identical inputs in both the typed |ParaLens|
implementation and a dynamic reference model, runs one gradient-descent
step via |runStep| with learning rate~1, and recovers the autograd
gradient as the difference between old and new parameters.  The typed
gradient is compared against this reference with a maximum absolute
tolerance of $10^{-4}$.  Because the typed backward pass is statically
compiled Haskell rather than a traced computation graph, any discrepancy
would indicate an error in the hand-derived Jacobian, not a numerical
accident.

\subsection*{Coverage}

\begin{itemize}
  \item \textbf{|linear|}: forward pass ($xW^\top$) and weight gradient
    ($\partial L/\partial W = \mathit{grad}^\top x$), batch size~1.
  \item \textbf{|addLens|}: forward pass (broadcast add) and bias
    gradient ($\partial L/\partial b = \sum_{\text{batch}} \mathit{grad}$),
    batch size~4.
  \item \textbf{|sigmoid|}: forward pass ($\sigma(x)$) and input gradient
    ($\mathit{grad} \cdot \sigma(x)(1-\sigma(x))$), 10 values.
  \item \textbf{|relu|}: forward pass and Heaviside input gradient, 10
    values.
  \item \textbf{|convLens|}: forward pass (unit stride, zero padding),
    and kernel gradient via the |im2col| batched matrix multiply, on a
    $1\times1\times7\times7$ input with a $2\times1\times3\times3$ kernel.
  \item \textbf{|flatten|}: forward reshape and backward reshape (both
    are pure shape operations with no arithmetic, verified to be exact
    inverses).
  \item \textbf{Iris forward equivalence}: the full two-layer Iris model
    run on a batch of eight samples produces output identical to an
    equivalent dynamic hasktorch model given the same weights.
  \item \textbf{Iris epoch equivalence}: after one full training epoch
    (18 batches), the typed and dynamic models reach identical parameter
    values.
\end{itemize}

% \noindent The suite does not currently cover the attention backward pass
% or the convolution input gradient ($\partial L/\partial x$); these
% remain directions for future testing.

\subsection*{Relationship to Accuracy}

Layer-level gradient tests are a stronger correctness signal than
end-to-end accuracy: a model can converge to a non-trivial accuracy
even with slightly incorrect gradients, whereas a Jacobian error above
$10^{-4}$ would be caught directly.  The epoch-equivalence test extends
this to the composed training loop, confirming that composition via
|(.#.)| and the |repara| update rule interact correctly across an
entire epoch on real data.
