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
module Optim where
\end{code}
%endif

\chapter{Optimisers}
\label{chap:optim}

A layer such as |matMulLensCore| knows how to compute gradients with
respect to its weights but not what to \emph{do} with them.  The
gradient update rule---gradient descent, momentum, Adam---is a separate
concern, and the framework keeps it separate.  Optimisers are lenses
plugged into the parameter slot of a layer via |repara|
(Section~\ref{sec:repara}), converting raw gradients into updated
parameters without touching the layer's forward or backward pass.

\section{Gradient Descent}
\label{sec:gradupdate}

The simplest update rule is plain gradient descent: add the incoming
gradient directly to the current parameter.  In a cartesian reverse
differential category this is the \emph{natural addition}
$\theta \leftarrow \theta + \partial\theta$; the sign of
$\partial\theta$ (controlled by the learning-rate cap) determines
whether the step is descent or ascent.  The implementation is a single
line:

\begin{code}
gradUpdate :: Num p => Lens' p p
gradUpdate = lens id (+)
\end{code}

\noindent The getter |id| reads the parameter unchanged; the setter
|(+)| adds the incoming gradient |p'| to the stored parameter |p|,
returning |p + p'|.  Equipping any layer with this rule is a one-liner:

\begin{code}
withGradDesc :: Num p => ParaLens p p a a' b b' -> ParaLens p p a a' b b'
withGradDesc = repara gradUpdate
\end{code}

\noindent The full gradient-descent training step is then:

\begin{code}
(withGradDesc net .#. loss) . lrSmoothT lr
  :: ParaLens' (net_params, target) input ()
\end{code}

\noindent |withGradDesc| ensures each backward pass computes
$\theta \leftarrow \theta + \partial\theta$; |lrSmoothT lr|
(Section~\ref{sec:lrcap}) provides the negative seed $\partial\theta
= -\eta \cdot \partial L / \partial\theta$ that makes the step descent.
Neither component knows about the other.

\section{Learning Rates as Caps}
\label{sec:lrcap}

The learning-rate family is formalised by a type synonym for lenses whose
output type is the unit:

\begin{code}
type LRLens l l' = Lens l l' () ()
\end{code}

\noindent A value of type |LRLens l l'| closes the output wire to the
monoidal unit $I = \mathtt{()}$ in both directions---exactly the cap
identified in Chapter~\ref{chap:loss}.  The general constructor is:

\begin{code}
learningRate :: (l -> l') -> LRLens l l'
learningRate alpha = lens (const ()) (const . alpha)
\end{code}

\noindent The getter |const ()| discards the scalar loss in the forward
direction.  The setter |const . alpha| ignores the upstream |()| and
applies |alpha| to the current learning-rate state |l|, emitting |alpha
l| as the gradient seed.  For a constant negative tensor learning rate:

\begin{code}
lrSmoothT lr = learningRate (const (negate (scalarT lr)))
\end{code}

\noindent where |scalarT| lifts a Haskell scalar to a rank-0 tensor
|Tensor dv dt []|.  Composing |lrSmoothT lr| on the right of
|net .#. loss| via plain |(.)| closes the last wire to unit and
seeds backpropagation with $-\eta$, as shown in
Section~\ref{sec:losscaps}.

\section{Momentum}
\label{sec:momentum}

Gradient descent converges slowly in the presence of high curvature.
Momentum methods accumulate a velocity term that damps oscillations and
accelerates convergence along low-curvature directions.  In the lens
framework, a momentum optimiser is a |Lens'| that \emph{expands} the
parameter type from |t shape| to |(t shape, t shape)|, carrying the
velocity buffer alongside the weights:

\begin{code}
type Momentum = Lens' (t shape, t shape) (t shape)
\end{code}

\noindent The state |(v, p)| flows on the vertical parameter wire: |p|
is the weight tensor read by the layer, and |v| is the velocity buffer
memorised between steps.

\subsection*{SGD with Momentum}

\begin{code}
momentum :: Scalar a => a -> Momentum
momentum = lens snd . momrev
\end{code}

\noindent The getter |snd| extracts the current weights |p| for the
layer.  The setter |momrev gamma (v, p) p'| computes:
\[
  v' = -\gamma v + \partial p, \qquad p_{\text{new}} = p + v',
\]
where $\partial p$ is the incoming gradient.  Setting $\partial p = -\eta\,\partial L/\partial p$ (from |lrSmoothT|) and expanding gives the standard SGD-with-momentum update.  The velocity decays by $\gamma$ each step and accumulates the gradient.

\subsection*{Nesterov Momentum}

\begin{code}
nesterov :: Scalar a => a -> Momentum
nesterov gamma = lens (uncurry fwd) (momrev gamma)
  where fwd v p = p + T.mulScalar gamma v
\end{code}

\noindent The only difference from plain momentum is the getter: instead
of returning $p$ it returns the lookahead position $p + \gamma v$, so
the gradient is evaluated at the next predicted iterate.  The setter
|momrev gamma| is unchanged.  Both |momentum| and |nesterov| plug into
any layer via |repara (momentum 0.9) layer|, which expands the parameter
type from |p| to |(v, p)| with the velocity buffer initialised
automatically.

\section{Adaptive Methods}
\label{sec:adaptive}

Adaptive methods maintain per-parameter statistics and scale the
effective learning rate accordingly.  They are again |Lens'| values that
expand the parameter type.

\subsection*{AdaGrad}

\begin{code}
adaGrad :: (Scalar a, Fractional a) => a -> Momentum
adaGrad eps = lens snd rev
  where
    rev (g, p) p' = (g', p + update * p')
      where
        g'     = g + p' * p'
        update = T.mulScalar eps . T.reciprocal
               . T.addScalar delta $ T.sqrt g'
\end{code}

\noindent The accumulator |g| stores the running sum of squared
gradients.  The per-parameter effective learning rate is
$\varepsilon / \sqrt{g' + \delta}$, which shrinks automatically for
parameters that receive large or frequent updates, and stays large for
infrequently-updated parameters.  The constant $\delta = 10^{-7}$
prevents division by zero.

\subsection*{Adam}

Adam maintains two exponentially decaying moment estimates:

\begin{code}
adam :: (Scalar a, Fractional a) => a -> a -> a -> Momentum2
adam beta1 beta2 eps = lens snd rev
  where
    rev ((m, v), p) p' = ((m', v'), p + T.mulScalar eps update)
      where
        m'     = T.mulScalar beta1 m + T.mulScalar (1 - beta1) p'
        v'     = T.mulScalar beta2 v + T.mulScalar (1 - beta2) (p' * p')
        update = m' / T.addScalar delta (T.sqrt v')
\end{code}

\noindent where |Momentum2 = Lens' ((t shape, t shape), t shape) (t shape)|
carries state |((m, v), p)|.  |m| is the first moment (exponential
moving average of gradients), |v| is the second moment (exponential
moving average of squared gradients), and $\varepsilon$ is the step
size.  The update $m' / \sqrt{v' + \delta}$ normalises the gradient by
its recent root-mean-square, reducing the effective learning rate for
high-variance parameters.

The deepened state type |((m, v), p)| vs |Momentum|'s |(v, p)| is
automatic: |repara (adam 0.9 0.999 1e-3) layer| expands the parameter
type to |((m, v), weights)| with no changes to the layer or the rest of
the network.

\section{Per-layer Optimisers}
\label{sec:perlayer}

The central payoff of the |repara| mechanism is that different layers
can use \emph{different optimisers} with no coordination between them:

\begin{code}
net  =   repara (momentum 0.9)    matMulLensCore
    .#.  repara (adaGrad 1e-2)    matMulLensCore
\end{code}

\noindent The combined parameter type is
$((v,p_1),\,(g,p_2))$---each layer carries its own independent optimiser
state, tracked automatically through the product type from |(.#.)|.  No
global optimiser object is needed; the type system enforces that each
layer's update rule stays local.

\begin{figure}[h]
\centering
\begin{tikzpicture}
  %% Outer bounding box
  % \draw[thick] (-6,-1.2) rectangle (6,4.0);

  %% ── Layer boxes (single box per layer) ──────────────────────────
  \node[draw, minimum width=1.8cm, minimum height=1.4cm] (L1) at (-3, 0) {$f_1$};
  \node[draw, minimum width=1.8cm, minimum height=1.4cm] (L2) at ( 3, 0) {$f_2$};

  %% ── Optimiser blocks ─────────────────────────────────────────────
  \node[draw, thick, fill=black!5, minimum width=2.0cm, minimum height=0.7cm,
        rounded corners=3pt] (mom) at (-3, 2.5) {\small$\mathit{mom}$};
  \node[draw, thick, fill=black!5, minimum width=2.0cm, minimum height=0.7cm,
        rounded corners=3pt] (ada) at ( 3, 2.5) {\small$\mathit{ada}$};

  %% ── L1 parameter wires ───────────────────────────────────────────
  \draw[wire] (-3.4, 3.85) -- node[left]  {$(v,p_1)$}   (-3.4, 2.85);
  \draw[wire] (-3.4, 2.15) -- node[left]  {$p_1$}        ($(L1.north)+(-0.4,0)$);
  \draw[wire] ($(L1.north)+(0.4,0)$) -- node[right] {$p_1'$}     (-2.6, 2.15);
  \draw[wire] (-2.6, 2.85) -- node[right] {$(v',p_1')$}  (-2.6, 3.85);

  %% ── L2 parameter wires ───────────────────────────────────────────
  \draw[wire] ( 2.6, 3.85) -- node[left]  {$(g,p_2)$}   ( 2.6, 2.85);
  \draw[wire] ( 2.6, 2.15) -- node[left]  {$p_2$}        ($(L2.north)+(-0.4,0)$);
  \draw[wire] ($(L2.north)+(0.4,0)$) -- node[right] {$p_2'$}     ( 3.4, 2.15);
  \draw[wire] ( 3.4, 2.85) -- node[right] {$(g',p_2')$}  ( 3.4, 3.85);

  %% ── Data wires ───────────────────────────────────────────────────
  \draw[wire] (-6, 0.3)  -- node[above] {$a$}  ($(L1.west)+(0, 0.3)$);
  \draw[wire] ($(L1.east)+(0, 0.3)$) -- node[above] {$b$} ($(L2.west)+(0, 0.3)$);
  \draw[wire] ($(L2.east)+(0, 0.3)$) -- node[above] {$c$} (6, 0.3);
  \draw[wire] ($(L1.west)+(0,-0.3)$) -- node[below] {$a'$} (-6, -0.3);
  \draw[wire] ($(L2.west)+(0,-0.3)$) -- node[below] {$b'$} ($(L1.east)+(0,-0.3)$);
  \draw[wire] (6, -0.3) -- node[below] {$c'$} ($(L2.east)+(0,-0.3)$);
\end{tikzpicture}
\caption{Two layers with independent per-layer optimisers.  Layer $f_1$
  uses SGD with momentum (state $(v,p_1)$); layer $f_2$ uses AdaGrad
  (state $(g,p_2)$).  Data flows forward on the upper wires
  ($a \to b \to c$) and backward on the lower wires
  ($c' \to b' \to a'$); the two parameter channels are structurally
  independent, carrying different state types on the vertical axis.}
\label{fig:perlayer}
\end{figure}

\noindent This contrasts with conventional frameworks such as PyTorch or
JAX, where the optimiser is a global object that must be explicitly
configured with which parameters belong to which update rule, and where
mixing rules across layers requires manual parameter grouping.  Here the
types enforce the separation: the outer parameter type
$((v,p_1),(g,p_2))$ is the product of two independent optimiser states,
and the composition rule |(.#.)| assembles it automatically from the
individual |repara| calls.
