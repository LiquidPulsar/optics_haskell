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
module Loss where
\end{code}
%endif

\chapter{Loss Functions}
\label{chap:loss}

A neural network assembled from |ParaLens'| values computes a typed
transformation from inputs to predictions, but it does not yet know
\emph{what} to optimise.  Loss functions supply this missing piece.
In the lens framework they are not a separate concept bolted on from
the outside; they are simply further |ParaLens'| values that can be
composed with the network in the ordinary way.  This chapter describes
their type, their role in the composed diagram, and the three concrete
instances provided by the library.

\section{Losses, Caps, and the Learning Rate}
\label{sec:losscaps}

Every loss function in this framework has the type

\begin{code}
ParaLens' (t shape) (t shape) (t [])
\end{code}

\noindent
Reading the three type arguments: the \emph{parameter} |p = t shape| is
the target tensor; the \emph{input} |a = t shape| is the network's
prediction; and the \emph{output} |b = t []| is a scalar: a
zero-dimensional tensor carrying a single floating-point number.

The loss reduces the network's prediction to a scalar, but it does not
yet close the diagram to the monoidal unit.  Composing a network |net|
with a loss |loss| via |(.#.)| gives a term whose output is still
|t []|, not the unit type:
\[
  \mathit{net} \mathbin{\bullet} \mathit{loss}
  \;:\;
  \text{|ParaLens' (net_params, t shape) (t input_shape) (t [])|}.
\]

The true \emph{cap}, the morphism that closes the final wire to the
monoidal unit $I = \mathtt{()}$, is the learning rate.  The library
formalises this as a type synonym:

\begin{code}
type LRLens l l' = Lens l l' () ()
\end{code}

\noindent |LRLens l l'| has |()| as \emph{both} output types: the
forward pass discards the scalar (outputting the unit), and the backward
pass emits the learning rate.  The general constructor is:

\begin{code}
learningRate :: (l -> l') -> LRLens l l'
learningRate alpha = lens (const ()) (const . alpha)
\end{code}

\noindent where |const . alpha| ignores the upstream |()| and applies
|alpha| to the current state |l| to produce the gradient seed |l'|.
For a constant tensor learning rate, |lrSmoothT| specialises this by
wrapping a Haskell scalar in a rank-0 tensor and negating it:

\begin{code}
lrSmoothT lr = learningRate (const (negate (scalarT lr)))
\end{code}

\noindent Composing with |(.)| closes the diagram fully:

\begin{code}
(withGradDesc net .#. loss) . lrSmoothT lr
  :: ParaLens' (net_params, target) input ()
\end{code}

\noindent The output type is now |()|, the monoidal unit, so no
information leaves the diagram in the forward direction.  The backward
pass flows $-\mathit{lr}$ back through every layer as the gradient seed,
updating all weight tensors in a single traversal.  The |withGradDesc|
wrapper (Chapter~\ref{chap:optim}) handles the CRDC natural addition
$p \leftarrow p + \partial p$; the cap supplies the sign and magnitude.

\section{The Gradient Scaling Convention}
\label{sec:lossscaling}

As established in Section~\ref{sec:crdc-bg}, the training loop
performs the CRDC natural addition
\[
  \theta_{\text{new}} \;=\; \theta_{\text{old}} + \partial\theta,
\]
where $\partial\theta$ is emitted by the backward pass; its sign and
magnitude are the sole determinants of whether the step is descent or
ascent.  The backward pass of the MSE loss (Section~\ref{sec:mse}) produces
\[
  \partial p \;=\; \tfrac{2}{N}\,\alpha \cdot (p - t),
\]
where |t| is the target, $N$ is the total number of output elements,
and |alpha| is the scalar |b'| received from above.  With
$\alpha = -\eta$ (a \emph{negative} learning rate), the update becomes
\[
  p_{\text{new}}
  \;=\; p_{\text{old}} + \tfrac{2}{N}(-\eta)(p - t)
  \;=\; p_{\text{old}} - \tfrac{2\eta}{N}(p - t),
\]
which is exact gradient descent on the mean-squared-error loss.  The
$\tfrac{2}{N}$ factor arises from differentiating $\tfrac{1}{N}\sum_i
(p_i - t_i)^2$; without it the update would correspond to a different
(unnormalised) loss, breaking equivalence with standard automatic
differentiation frameworks.  The negative sign is not a convention
trick; it is the mechanism by which the additive CRDC structure produces
descent rather than ascent.  Choosing $\alpha > 0$ gives ascent; the library includes |deepDreamLoss| (a
dot-product loss $\langle t, p\rangle$ with symmetric gradients) to support
activation-maximisation settings without any change to the training loop.

\section{Mean-Squared Error: \texttt{lossSmooth}}
\label{sec:mse}

The MSE loss is a |ParaLens'| over tensors of arbitrary |shape|:

\begin{code}
lossSmooth
  ::  ( TrivialFacts shape
      , T.BasicArithmeticDTypeIsValid dv dt
      , T.StandardFloatingPointDTypeValidation dv dt )
  =>  ParaLens' (t shape) (t shape) (t [])
\end{code}

\noindent The constraint alias |TrivialFacts shape| captures two
hasktorch API requirements

\begin{code}
type TrivialFacts shape =
  ( shape ~ T.Broadcast shape shape
  , shape ~ T.Reverse (T.Reverse shape) )
\end{code}

---that are semantically trivial (every shape satisfies them)
but must be supplied as explicit witnesses to satisfy the type-checker
when calling |T.sub| and |T.mul|.

The implementation is:

\begin{code}
lossSmooth = lens fwd (flip rev')
  where
    fwd              = uncurry $ T.mseLoss @T.ReduceMean
    rev' alpha       = (id &&& T.neg) . T.mul alpha . uncurry T.sub
                       -- fans out gradient to both outputs: (d, -d)
\end{code}

\noindent The forward pass delegates to |T.mseLoss @T.ReduceMean|, which
computes $\tfrac{1}{n}\sum_i(t_i - p_i)^2$ averaged over all elements.
The backward pass applies the chain rule: with $N$ elements and incoming
scalar $\alpha$ from the learning-rate cap,
\[
  \partial t = \tfrac{2}{N}\,\alpha\,(t - p),
  \qquad
  \partial p = -\tfrac{2}{N}\,\alpha\,(t - p).
\]

\section{Softmax Cross-Entropy: \texttt{softMaxCELoss}}
\label{sec:celoss}

For classification tasks the cross-entropy loss fuses the softmax into
the loss for numerical stability:

\begin{code}
softMaxCELoss
  ::  ( T.All KnownNat [x, c]
      , T.BasicArithmeticDTypeIsValid dv dt
      , T.StandardFloatingPointDTypeValidation dv dt
      , T.SumDTypeIsValid dv dt, T.MeanDTypeValidation dv dt )
  =>  ParaLens' (t [x, c]) (t [x, c]) (t [])
\end{code}

\noindent The parameter is a distribution over |c| classes for each of
|x| examples (one-hot or soft labels); the input is the corresponding
logit tensor.

\begin{code}
softMaxCELoss = lens fwd rev
  where
    fwd (bt, bp)  = negate . T.meanAll . T.sumDim @1
                    $ bt * T.logSoftmax @1 bp
    rev (bt, bp) d  = (T.mul d $ negate $ T.log q, T.mul d $ q - bt)
      where q       = T.softmax @1 bp
\end{code}

\noindent The forward pass computes
\[
  -\frac{1}{x}\sum_{i=1}^{x}\sum_{c=1}^{C} t_{ic}\log q_{ic},
  \qquad q = \operatorname{softmax}(p),
\]
the standard categorical cross-entropy averaged over the batch.
Throughout, the |@1| type application selects axis~1 (class
dimension) so that softmax and the summation operate over classes
while leaving axis~0 (the batch) intact.

The backward pass exploits the well-known simplification that arises
when softmax and cross-entropy are fused: the gradient with respect to
the logits is simply $d\,(q - t)$.  The gradient with respect to the
targets, $-d\log q$, is available because the lens type treats both
arguments symmetrically; in practice the training loop discards it, but
it is useful in meta-learning settings where the targets themselves are
learnable.
